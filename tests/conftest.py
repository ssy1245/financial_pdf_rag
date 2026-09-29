"""注册 --run-dense 开关，实际模型测试默认跳过。
"""
def pytest_addoption(parser):
    parser.addoption(
        "--run-dense", action="store_true", default=False,
        help="运行本地 Embedding 模型的 8 题 Dense 检索测试（可能下载模型）",
    )
