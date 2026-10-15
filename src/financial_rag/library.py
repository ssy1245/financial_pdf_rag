"""Atomic local library snapshots, isolated by the signed browser session.

The caller serializes writes with the app lock. This is a single-process store.
A complete snapshot is published by replacing one small current pointer.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import tempfile


class LibraryError(RuntimeError):
    pass


class SavedFolder:
    def __init__(self, path):
        self.name = str(path)

    def cleanup(self):
        # Explicit cleanup only: garbage collection must never remove saved data.
        shutil.rmtree(self.name, ignore_errors=True)


def session_secret(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / 'session.key'
    try:
        with path.open('x', encoding='ascii') as f:
            os.chmod(path, 0o600)
            f.write(secrets.token_hex(32))
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        pass
    key = path.read_text().strip()
    if not re.fullmatch(r'[a-f0-9]{64}', key):
        raise LibraryError('本地会话密钥损坏；请从备份恢复，勿覆盖已有资料库')
    return key


class LibraryStore:
    def __init__(self, root):
        self.root = Path(root)

    def directory(self, session_id):
        if not isinstance(session_id, str) or not session_id:
            raise LibraryError('缺少浏览器会话')
        return self.root / hashlib.sha256(session_id.encode()).hexdigest()

    def read(self, session_id):
        if not session_id:
            return None
        base = self.directory(session_id)
        pointer = base / 'current.json'
        if not pointer.exists():
            return None
        try:
            current = json.loads(pointer.read_text())['snapshot']
            if current is None:
                return None
            if not re.fullmatch(r'[a-f0-9]{32}', current):
                raise ValueError('invalid snapshot')
            folder = base / current
            manifest = json.loads((folder / 'manifest.json').read_text())
            if manifest['version'] != 1:
                raise ValueError('unsupported library format')
            docs = manifest['documents']
            if not docs or len({d['document_id'] for d in docs}) != len(docs):
                raise ValueError('invalid documents')
            for d in docs:
                key = d['document_id']
                suffix = Path(d['name']).suffix.lower()
                if not re.fullmatch(r'[a-f0-9]{64}', key) or suffix not in {'.pdf', '.docx', '.xlsx'}:
                    raise ValueError('invalid document')
                for name in (key+suffix, key+'.json'):
                    if not (folder / name).is_file():
                        raise ValueError('missing source')
            if not (folder / 'vectors.sqlite3').is_file():
                raise ValueError('missing index')
            return dict(folder=SavedFolder(folder), documents=docs,
                        names={d['document_id']:d['name'] for d in docs})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise LibraryError('已保存资料库无法读取；原文件保留，请检查磁盘或恢复备份') from exc

    def _publish(self, base, snapshot):
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(prefix='.pointer-', dir=base)
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump({'snapshot':snapshot}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, base / 'current.json')
        finally:
            Path(tmp).unlink(missing_ok=True)

    def commit(self, session_id, state):
        base = self.directory(session_id)
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
        folder = base / secrets.token_hex(16)
        try:
            shutil.copytree(state['folder'].name, folder)
            (folder / 'manifest.json').write_text(json.dumps({
                'version':1, 'documents':state['documents'],
            }, ensure_ascii=False, indent=2))
            # Flush complete snapshot before switching the visible version.
            for path in folder.iterdir():
                if path.is_file():
                    with path.open('rb') as f:
                        os.fsync(f.fileno())
            self._publish(base, folder.name)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        staging = state['folder']
        state['folder'] = SavedFolder(folder)
        try:
            staging.cleanup()
        except OSError:
            pass  # Published snapshot is already authoritative.
        self._prune(base, folder.name)

    def _prune(self, base, keep=None):
        try:
            paths = list(base.iterdir())
        except OSError:
            return  # Best-effort garbage collection must not undo a committed version.
        for path in paths:
            if path.is_dir() and re.fullmatch(r'[a-f0-9]{32}', path.name) and path.name != keep:
                shutil.rmtree(path, ignore_errors=True)

    def clear(self, session_id):
        if not session_id:
            return
        base = self.directory(session_id)
        self._publish(base, None)
        self._prune(base)
