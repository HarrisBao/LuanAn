from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorRetriever:
    """
    Bộ truy xuất tài liệu dạng đơn giản bằng TF-IDF.

    Mục tiêu:
    - Không lưu vector database lớn lên GitHub.
    - Dễ chạy demo.
    - Có thể thay thế bằng FAISS/Chroma/SentenceTransformer sau.
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = None

    def load_data(self) -> None:
        """
        Đọc dữ liệu tài liệu từ file CSV.

        File CSV nên có ít nhất các cột:
        - title
        - content
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {self.data_path}")

        self.df = pd.read_csv(self.data_path)

        required_columns = {"title", "content"}
        missing_columns = required_columns - set(self.df.columns)

        if missing_columns:
            raise ValueError(f"Thiếu cột bắt buộc trong CSV: {missing_columns}")

        self.df["content"] = self.df["content"].fillna("")
        self.df["title"] = self.df["title"].fillna("")

    def build_index(self) -> None:
        """
        Tạo vector index trong RAM.
        Không ghi file index lớn ra ổ đĩa.
        """
        if self.df is None:
            self.load_data()

        documents = (
            self.df["title"].astype(str) + " " + self.df["content"].astype(str)
        ).tolist()

        self.document_vectors = self.vectorizer.fit_transform(documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Tìm top_k tài liệu liên quan nhất với câu hỏi.
        """
        if self.document_vectors is None:
            self.build_index()

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.document_vectors).flatten()

        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]

            results.append(
                {
                    "rank": len(results) + 1,
                    "title": row["title"],
                    "content": row["content"],
                    "score": float(scores[idx]),
                }
            )

        return results


if __name__ == "__main__":
    retriever = VectorRetriever(data_path="data/sample/sample_documents.csv")
    retriever.build_index()

    query = "Tài liệu học Python cho người mới bắt đầu"
    results = retriever.search(query, top_k=3)

    for item in results:
        print(f"[{item['rank']}] {item['title']} - score={item['score']:.4f}")