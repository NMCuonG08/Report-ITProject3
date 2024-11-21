from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
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
embedding_model = OllamaEmbeddings(model="jina/jina-embeddings-v2-base-en")

# Tạo embedding cho các chunk
chunk_embeddings = [embedding_model.embed_documents([chunk['content']])[0] for chunk in chunks_with_titles]

# Truy vấn (câu tìm kiếm)
query = "Sinh Viên đã hoàn tất 150 TC của Chương Trình Đào Tạo có thể tốt nghiệp chưa?"

# Tạo embedding cho câu truy vấn
query_embedding = embedding_model.embed_query(query)

# Tính cosine similarity giữa query và từng chunk
similarities = []
for idx, chunk in enumerate(chunks_with_titles):
    chunk_embedding = chunk_embeddings[idx]
    similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
    similarities.append({'title': chunk['title'], 'content': chunk['content'], 'similarity': similarity})

# Sắp xếp kết quả theo mức độ similarity (từ cao đến thấp)
similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)

# Hiển thị kết quả
print("Kết quả tìm kiếm:")
for result in similarities[:30]:  # Hiển thị 30 kết quả tương đồng nhất
    print(f"Title: {result['title']}")
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Chunk: \n{textwrap.fill(result['content'], width=100)}\n")
    print("=" * 100)
