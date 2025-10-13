# DoAnTotNghiep
Đồ Án Tốt Nghiệp
## 🧱 Cấu trúc Dự án

Dưới đây là cấu trúc thư mục chính của dự án:

```text
├── .gitignore
├── README.md
├── requirements.txt
├── configs/
│   └── phobert_config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_prototyping.ipynb
├── saved_models/
├── results/
│   ├── figures/
│   └── metrics/
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── model.py
    ├── train.py
    └── utils.py
```

## 🧾 Mô tả Dataset

Dữ liệu được sử dụng trong dự án bao gồm các thông tin về tin tức thật và giả.  
Bảng dưới đây mô tả chi tiết các cột trong tập dữ liệu:

| **Tên Cột** | **Loại Dữ Liệu** | **Mục Đích** |
|--------------|------------------|---------------|
| `title` | Văn bản *(string)* | Tiêu đề của tin tức. |
| `text` | Văn bản *(string)* | Nội dung đầy đủ của tin tức. |
| `subject` | Phân loại *(string)* | Chủ đề của tin tức (ví dụ: `politicsNews`, `worldnews`). |
| `date` | Văn bản *(string)* | Ngày đăng tin. |
| `label` | Phân loại *(int64)* | Biến mục tiêu *(Target)*: `0` (tin giả) hoặc `1` (tin thật). |