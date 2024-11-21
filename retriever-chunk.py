from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import textwrap

# Đọc nội dung từ file văn bản
with open('document.txt', 'r', encoding='utf-8') as file:
    content = file.read()

chunk_size = 400  # Số ký tự mỗi chunk
chunk_overlap = 200  # Số ký tự chồng lặp giữa các chunk

# Hàm để chia văn bản theo tiêu đề trang (từ khóa "Page")
def split_text_by_page(text):
    pages = text.split("Page")  # Tách văn bản thành các trang dựa trên từ khóa "Page"
    return [page.strip() for page in pages if page.strip()]  # Loại bỏ khoảng trắng và trang rỗng

# Chia văn bản theo trang
pages = split_text_by_page(content)

# Sử dụng CharacterTextSplitter để chia từng trang thành các chunk nhỏ
text_splitter = CharacterTextSplitter(
    separator=" ",  # Tách văn bản dựa trên dấu cách
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

chunks_with_titles = []
for page_index, page_content in enumerate(pages):
    page_title = f"Page {page_index + 1}"  # Gắn tiêu đề cho từng trang
    chunks = text_splitter.split_text(page_content)  # Chia trang thành các chunk nhỏ
    for chunk in chunks:
        chunks_with_titles.append({"title": page_title, "content": chunk})

# Khởi tạo mô hình embedding
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

# Khởi tạo danh sách Document từ chunks_with_titles
documents = [
    Document(page_content=chunk["content"], metadata={"title": chunk["title"]})
    for chunk in chunks_with_titles
]

# Tạo FAISS index và thêm các embeddings
faiss_index = FAISS.from_documents(documents, embedding_model)

# Truy vấn (câu tìm kiếm)
query = "Sinh Viên đã hoàn tất 150 TC của Chương Trình Đào Tạo có thể tốt nghiệp chưa?"

# Tìm kiếm các kết quả liên quan bằng retriever
retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 30})
results = retriever.get_relevant_documents(query)

# Hiển thị kết quả trực tiếp từ retriever
print("Kết quả tìm kiếm:")
for result in results:
    print(f"Title: {result.metadata['title']}")  # Tiêu đề trang
    print(f"Chunk: \n{textwrap.fill(result.page_content, width=100)}\n")
    print("=" * 100)
