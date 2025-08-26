# Fashion Recommendation System 

 - Github: [ktoan911](https://github.com/ktoan911)
- Email: khanhtoan.forwork@gmail.com

The system is based on the **FashionVLP** model introduced in the paper [FashionVLP: Vision Language Transformer for Fashion Retrieval with Feedback](https://openaccess.thecvf.com/content/CVPR2022/papers/Goenka_FashionVLP_Vision_Language_Transformer_for_Fashion_Retrieval_With_Feedback_CVPR_2022_paper.pdf), with minor modifications in the architecture by removing the clothing part extraction layers. This adjustment improves training and inference speed while maintaining comparable accuracy to the original model.


---

## 📌 Overview

FashionVLP tackles the task of **fashion image retrieval with textual feedback**:  
- Input: a **reference image** and a **text feedback** (e.g., *"longer sleeves, lighter color"*).  
- Output: retrieve **target images** that best match the modified description.  

<p align="center">
  <img src="./Assets/vlp.png" width="750"/>
</p>

---

## ⚙️ Setting Up the Environment
#### Step 1: Create a Conda environment named your_env_name with Python 3.9.*

```python
conda create -n ${your_env_name} python= 3.9.*
conda activate ${your_env_name}
```

#### Step 2: Install the packages from the requirements.txt file

```
pip install -r requirements.txt
```

#### Step 3: Create a .env file and add the following lines, replacing the placeholders with your actual values:
```env
MONGO_URI=
DB_NAME= 
COLLECTION_NAME=
```

#### Step 4: Test model with Streamlit:
```env
bash run_app.sh
```
---

## 📂 Dataset

We support the same datasets as the paper **[FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq)**  

Prepare datasets under `data/` with the following structure:

```
datasets/
  ├── fashionIQ/
        ├── images/
        ├── train.json
        └── val.json

```

---

## 🚀 Training

Train on **FashionIQ**:

```bash
bash train.sh <annotation_file_path> <train_image_folder_path> <batch_size> <epochs>
```

Excample: 
```bash
bash train.sh data/annotation_file_path.json data/images 16 10
```
---

## 🏗️ Model Architecture

- **Reference block**:  
  - Extracts multi-level features:  
    - Whole image (`f_whole`)  
    - Cropped image (`f_crop`)  
    - Fashion landmarks (`f_lm`)  
  - Fused with text tokens via **BERT-based transformer** → produces joint embedding `f_ref`.

- **Target block**:  
  - Uses same features (`f_whole`, `f_crop`, `f_lm`)  
  - Combines them via **positional + landmark attention**  
  - Projects to joint space → embedding `f_tar`.

- **Training**:  
  - Similarity = cosine(`f_ref`, `f_tar`)  
  - Loss = batch classification loss (contrastive).

