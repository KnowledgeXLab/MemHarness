#!/usr/bin/env python3
"""
测试 vLLM 部署的 embedding 模型服务是否正常运行
用法: python test_embedding.py --url http://localhost:8000/v1/embeddings --model your-model-name
"""

import argparse
import sys
import requests
import json


def test_embedding(url: str, model: str, test_text: str = "这是一个测试文本"):
    """
    测试 embedding 服务

    Args:
        url: embedding 服务 URL，例如 http://localhost:8000/v1/embeddings
        model: 模型名称
        test_text: 用于测试的文本

    Returns:
        bool: 测试是否成功
    """
    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": test_text,
    }

    print(f"\n{'='*60}")
    print(f"测试 vLLM Embedding 服务")
    print(f"{'='*60}")
    print(f"请求 URL: {url}")
    print(f"模型名称: {model}")
    print(f"测试文本: {test_text}")
    print(f"{'='*60}\n")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ 请求成功!")

            # 解析响应
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0].get("embedding", [])
                embedding_dim = len(embedding)
                print(f"Embedding 维度: {embedding_dim}")
                print(f"Embedding 前5个值: {embedding[:5]}")

                # 显示 token 使用情况（如果有）
                if "usage" in result:
                    usage = result["usage"]
                    print(f"\nToken 使用情况:")
                    print(f"  - prompt_tokens: {usage.get('prompt_tokens', 'N/A')}")
                    print(f"  - total_tokens: {usage.get('total_tokens', 'N/A')}")

                print(f"\n{'='*60}")
                print(f"✅ 测试通过！Embedding 服务运行正常")
                print(f"{'='*60}")
                return True
            else:
                print(f"⚠️ 响应格式异常，缺少 'data' 字段")
                print(f"响应内容: {json.dumps(result, indent=2, ensure_ascii=False)}")
                return False
        else:
            print(f"\n❌ 请求失败!")
            print(f"错误码: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
            except:
                print(f"响应内容: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"\n❌ 连接失败!")
        print(f"无法连接到 {url}")
        print(f"请检查: 1) 服务是否已启动  2) URL 地址是否正确")
        return False

    except requests.exceptions.Timeout:
        print(f"\n❌ 请求超时!")
        print(f"服务响应时间超过 30 秒")
        return False

    except Exception as e:
        print(f"\n❌ 发生异常!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="测试 vLLM 部署的 embedding 模型服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试本地服务
  python test_embedding.py --url http://localhost:8000/v1/embeddings --model BAAI/bge-large-zh-v1.5

  # 测试远程服务
  python test_embedding.py --url http://192.168.1.100:8000/v1/embeddings --model bge-m3

  # 使用自定义测试文本
  python test_embedding.py --url http://localhost:8000/v1/embeddings --model bge-large --text "自定义测试文本"
        """
    )

    parser.add_argument(
        "--url",
        type=str,
        default="http://10.140.37.68:8081/v1/embeddings",
        help="embedding 服务 URL (默认: http://localhost:8000/v1/embeddings)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="bge_m3",
        help="模型名称 (例如: BAAI/bge-large-zh-v1.5, bge-m3)"
    )

    parser.add_argument(
        "--text",
        type=str,
        default="这是一个测试文本",
        help="用于测试的文本内容"
    )

    args = parser.parse_args()

    success = test_embedding(args.url, args.model, args.text)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
