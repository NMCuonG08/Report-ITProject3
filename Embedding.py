from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import OllamaEmbeddings
import textwrap

# Đọc nội dung từ file văn bản
with open('document.txt', 'r', encoding='utf-8') as file:
    content = file.read()

Chunk_size=100

# Hàm để chia văn bản thành các chunk với tên trang
def chunk_text_with_page_titles(text, chunk_size=Chunk_size):
    pages = text.split("Page")
    chunks_with_titles = []

    for page_index, page_content in enumerate(pages):
        page_title = f"Page {page_index}"
        words = page_content.split()

        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks_with_titles.append({"title": page_title, "content": chunk})

    return chunks_with_titles


# Chia văn bản thành các chunk với tên trang
chunks_with_titles = chunk_text_with_page_titles(content, chunk_size=Chunk_size)

# Khởi tạo mô hình embedding
embedding_model = OllamaEmbeddings(model="snowflake-arctic-embed")

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
for result in similarities[:30]:  # Hiển thị 5 kết quả tương đồng nhất
    print(f"Title: {result['title']}")
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Chunk: \n{textwrap.fill(result['content'], width=100)}\n")
    print("=" * 100)
