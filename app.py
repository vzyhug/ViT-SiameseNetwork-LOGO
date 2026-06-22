import os
import torch
import faiss
import pandas as pd
import streamlit as st
from PIL import Image
from torchvision import transforms
from transformers import ViTModel
import torch.nn as nn

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Hệ thống nhận diện Logo", page_icon="🔍", layout="centered")

class ViTEncoder(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224'):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
    def forward(self, pixel_values):
        return self.vit(pixel_values=pixel_values).pooler_output

class SiameseNetwork(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward_once(self, x):
        return self.backbone(x)

# Cache dữ liệu và model để tránh tải lại mỗi lần app re-run
@st.cache_resource
def load_models_and_data():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    backbone = ViTEncoder()
    model = SiameseNetwork(backbone)
    
    # Load weights
    model.load_state_dict(torch.load('models/best_model_logo2k.pth', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Load FAISS index & metadata
    index = faiss.read_index('models/faiss_hnsw.index')
    metadata = pd.read_csv('models/metadata_logo2k.csv')
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return model, index, metadata, transform, DEVICE

# Gọi hàm load dữ liệu
try:
    model, index, metadata, transform, DEVICE = load_models_and_data()
except Exception as e:
    st.error(f"Lỗi khi tải model hoặc dữ liệu: {e}")
    st.stop()

# Xây dựng giao diện chính
st.title("🔍 Truy Xuất Logo (ViT + FAISS)")
st.markdown("Hệ thống nhận diện logo sử dụng **Vision Transformer** và tìm kiếm vector bằng **FAISS**.")

uploaded_file = st.file_uploader("Kéo thả ảnh hoặc click để tải lên", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh upload
    img = Image.open(uploaded_file)
    st.image(img, caption="Ảnh đã tải lên", width=300)
    
    with st.spinner('Đang xử lý và tìm kiếm độ tương đồng...'):
        try:
            # Tiền xử lý an toàn
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGBA')
                background = Image.new('RGBA', img.size, (255, 255, 255))
                background.paste(img, mask=img)
                img = background.convert('RGB')
            else:
                img = img.convert('RGB')
                
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            # Rút trích đặc trưng
            with torch.no_grad():
                emb = model.backbone(img_tensor).cpu().float().numpy().flatten().astype('float32').reshape(1, -1)
                    
            # Tìm kiếm với FAISS
            faiss.normalize_L2(emb)
            scores, indices = index.search(emb, 5)
            
            st.subheader("🚀 Kết quả tương đồng hàng đầu:")
            
            # Tạo 5 cột để hiển thị 5 ảnh cạnh nhau
            cols = st.columns(5)
            
            for i, idx in enumerate(indices[0]):
                logo_id = metadata.iloc[idx]['logo_id']
                filename = metadata.iloc[idx]['filename']
                similarity_pct = max(0.0, float(scores[0][i]) * 100.0)
                
                # Đường dẫn ảnh thực tế trên ổ cứng (Nằm trong thư mục data)
                img_path = os.path.join("data", filename)
                
                with cols[i]:
                    # Hiển thị ảnh nếu file tồn tại
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                    else:
                        st.error("Không tìm thấy ảnh")
                        
                    # Hiển thị ID và phần trăm tương đồng
                    st.markdown(f"**{logo_id}**")
                    if similarity_pct > 80:
                        st.success(f"{similarity_pct:.1f}%")
                    elif similarity_pct > 50:
                        st.warning(f"{similarity_pct:.1f}%")
                    else:
                        st.error(f"{similarity_pct:.1f}%")
                
        except Exception as e:
            st.error(f"Đã xảy ra lỗi hệ thống trong quá trình xử lý: {e}")