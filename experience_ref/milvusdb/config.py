from datetime import datetime
from typing import List
import os
import logging
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from openai import OpenAI
from pymilvus import MilvusClient
import shutil
import time
import httpx
import json

# -------- 配置日志 --------
# Allow overriding paths via environment variables for experiment-specific data
BASE_DIR = os.environ.get("VDB_BASE_DIR", "./vdb_default")
os.makedirs(BASE_DIR, exist_ok=True)

LOG_FILE = os.path.join(BASE_DIR, "db_server.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# -------- ExpConfig --------
class ExpConfig:
    MILVUS_DB_PATH = os.path.join(BASE_DIR, "milvus_exp.db")
    MAX_SUCCESS_TRAJECTORIES_PER_PRINCIPLE = 3
    MAX_FAILED_TRAJECTORIES_PER_PRINCIPLE = 2
    
    @staticmethod
    def get_collection_names(experiment_name: str = None):
        """根据实验名获取集合名称，如果未提供实验名则使用时间戳"""
        if experiment_name:
            principles_name = f"exp_principles_{experiment_name}"
            trajectories_name = f"exp_trajectories_{experiment_name}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            principles_name = f"exp_principles_{timestamp}"
            trajectories_name = f"exp_trajectories_{timestamp}"
        
        return principles_name, trajectories_name
    
    MILVUS_COLLECTION_NAME, MILVUS_TRAJECTORY_COLLECTION_NAME = get_collection_names.__func__(None)

exp_config = ExpConfig()

logger.info(f"VDB_BASE_DIR is set to: {os.path.abspath(BASE_DIR)}")
logger.info(f"Milvus DB path is: {os.path.abspath(exp_config.MILVUS_DB_PATH)}")


# -------- EmbeddingProvider 通过远程 API 获取 embedding --------
class EmbeddingProvider:
    def __init__(self, api_url: str = "http://10.140.37.64:8000/v1", api_key: str = "empty", model_name: str = "bge_m3"):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name

        # 避免受容器内 SSL 相关环境变量影响
        os.environ.pop("SSL_CERT_FILE", None)
        os.environ.pop("SSL_CERT_DIR", None)

        # 为 OpenAI SDK 构造自定义 httpx 客户端，禁用环境信任与证书校验（内网 http）
        try:
            http_client = httpx.Client(verify=False, trust_env=False, timeout=30.0)
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_url, http_client=http_client)
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client, fallback to zero embeddings. Error: {e}")
            self.client = None

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        try:
            embedding_obj = self.client.embeddings.create(input=texts, model=self.model_name)
            embeddings = [d.embedding for d in embedding_obj.data]
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"[EmbeddingProvider ERROR] {e}")
            return np.zeros((len(texts), 1024))  # fallback

# -------- 数据库清理函数 --------
def cleanup_db_files(db_path: str):
    """清理数据库文件和相关目录"""
    try:
        if os.path.exists(db_path):
            logger.warning(f"Removing existing database file: {db_path}")
            if os.path.isfile(db_path):
                os.remove(db_path)
            elif os.path.isdir(db_path):
                shutil.rmtree(db_path)
        
        # 清理相关的目录和文件
        db_dir = os.path.dirname(db_path)
        for item in os.listdir(db_dir):
            item_path = os.path.join(db_dir, item)
            if item.startswith("milvus") and (item.endswith(".db") or item.endswith(".wal") or item.endswith(".log")):
                logger.warning(f"Removing milvus related file: {item_path}")
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    
    except Exception as e:
        logger.error(f"Error cleaning up database files: {e}")

def check_db_directory_permissions(db_path: str):
    """检查数据库目录的权限"""
    db_dir = os.path.dirname(db_path)
    try:
        # 检查目录是否存在，不存在则创建
        os.makedirs(db_dir, exist_ok=True)
        
        # 检查写权限
        test_file = os.path.join(db_dir, "test_write_permission.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        
        logger.info(f"Database directory permissions OK: {db_dir}")
        return True
    except Exception as e:
        logger.error(f"Database directory permission error: {e}")
        return False

# -------- 基础向量数据库客户端 --------
class BaseVectorDBClient:
    def __init__(self, db_path: str, api_url: str = "http://10.140.37.64:8000/v1", api_key:str = "empty", model_name: str = "bge_m3", max_retries: int = 3):
        self.db_path = db_path
        self.embedding_provider = EmbeddingProvider(api_url=api_url, api_key=api_key, model_name=model_name)
        self.dim = 1024  # 固定为 bge-m3 的输出维度
        self.max_retries = max_retries
        
        logger.info(f"[BaseVectorDBClient] Connecting to embedding API at {api_url}, dim={self.dim}, model={model_name}")
        logger.info(f"[BaseVectorDBClient] Database path: {db_path}")
        
        # 初始化 Milvus 客户端
        self.client = self._create_milvus_client()

    def _create_milvus_client(self) -> MilvusClient:
        """创建 Milvus 客户端，带重试机制"""
        # 检查目录权限
        if not check_db_directory_permissions(self.db_path):
            raise Exception("Database directory permission check failed")
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempting to create Milvus client (attempt {attempt + 1}/{self.max_retries})...")
                client = MilvusClient(self.db_path)
                logger.info("Milvus client created successfully")
                return client
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # 尝试清理数据库文件
                    logger.warning("Attempting to clean up database files...")
                    cleanup_db_files(self.db_path)
                    # 等待一段时间再重试
                    time.sleep(2)
                else:
                    logger.error("All attempts failed. Consider:")
                    logger.error("1. Check disk space")
                    logger.error("2. Check directory permissions")
                    logger.error("3. Kill any existing milvus processes")
                    logger.error("4. Try a different database path")
                    raise Exception(f"Failed to create Milvus client after {self.max_retries} attempts: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        ''' 将单个文本转换为 embedding 向量。 '''
        if not text:
            return [0.0] * self.dim
        return self.embedding_provider.get_embeddings([text])[0].tolist()

    def drop_collection(self, collection_name: str):
        """删除指定集合"""
        try:
            if collection_name in self.client.list_collections():
                self.client.drop_collection(collection_name)
                logger.info(f"Dropped collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error dropping collection {collection_name}: {e}")
