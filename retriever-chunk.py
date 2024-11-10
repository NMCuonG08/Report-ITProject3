from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
import textwrap
import nltk
import re

# Đảm bảo nltk có thể tải xuống các tài nguyên cần thiết cho việc phân câu
nltk.download('punkt')

# Đọc nội dung từ file văn bản
with open('document.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# Hàm để chia văn bản thành các câu với tên trang, chia thành nhóm 3 câu
def chunk_text_with_page_titles(text, sentences_per_chunk=2):
    pages = text.split("Page")
    chunks_with_titles = []

    # Regular expression pattern for sentence splitting (period, question mark, exclamation mark, semicolon)
    sentence_endings = re.compile(r'(?<=[.!?;])\s+')

    for page_index, page_content in enumerate(pages):
        page_title = f"Page {page_index}"

        # Split content based on the sentence-ending punctuation pattern
        sentences = sentence_endings.split(page_content.strip())

        # Chia thành các chunk có 2 câu mỗi chunk
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = " ".join(sentences[i:i + sentences_per_chunk])
            chunks_with_titles.append({"title": page_title, "content": chunk})

    return chunks_with_titles

# Chia văn bản thành các chunk (mỗi chunk có 2 câu)
chunks_with_titles = chunk_text_with_page_titles(content, sentences_per_chunk=1)

# Khởi tạo mô hình embedding và retriever
embedding_model = OllamaEmbeddings(model="jina/jina-embeddings-v2-base-en")

# Tạo embeddings cho các chunk và khởi tạo kho lưu trữ với Chroma vectorstore
texts = [chunk['content'] for chunk in chunks_with_titles]
metadata = [{"title": chunk['title']} for chunk in chunks_with_titles]

# Index các chunk bằng cách sử dụng Chroma vectorstore làm kho lưu trữ
vectorstore = Chroma.from_texts(texts, embedding_model, metadatas=metadata)

# Khởi tạo retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 30})

# Truy vấn (câu tìm kiếm)
query = "Sinh Viên đã hoàn tất 150 TC của Chương Trình Đào Tạo có thể tốt nghiệp chưa?"

# Lấy kết quả từ retriever
results = retriever.get_relevant_documents(query)

# Hiển thị kết quả
print("Kết quả tìm kiếm:")
for result in results:
    print(f"Title: {result.metadata['title']}")
    print(f"Chunk: \n{textwrap.fill(result.page_content, width=100)}\n")
    print("=" * 100)
