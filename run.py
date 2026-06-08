#!/usr/bin/env python3
import os
import sys
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

def main():
    print("=" * 50)
    print("  AI智能用户召回系统 - 启动中")
    print("=" * 50)

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))

    print(f"服务地址: http://{host}:{port}")
    print("请在浏览器中打开上述地址访问运营后台")
    print()

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
