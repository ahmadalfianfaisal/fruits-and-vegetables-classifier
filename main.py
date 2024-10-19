import torch
from torchvision import models, transforms
from PIL import Image
import streamlit as st
from huggingface_hub import hf_hub_download
import requests

st.set_page_config(
    layout='wide',
    page_title="Fruit and Vegetable Classifier", 
    initial_sidebar_state='expanded'
)



def load_model():
    model_path = hf_hub_download(repo_id="ahmadalfian/fruits_vegetables_classifier", filename="resnet50_finetuned.pth")
    model = models.resnet50(pretrained=False)
    num_classes = 36
    model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


# Fungsi untuk memproses gambar
def process_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(image)
    return img_tensor.unsqueeze(0)


def classify_image(model, image):
    img_tensor = process_image(image)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()


def get_nutrition_info(food_name):
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"

    params = {
        "query": food_name,
        "pageSize": 1,
        "api_key": api_key
    }

    response = requests.get(url, params=params)

    data = response.json()

    if "foods" in data and len(data["foods"]) > 0:
        food = data["foods"][0]
        nutrients_totals = {
            "Energy": 0,
            "Carbohydrate, by difference": 0,
            "Fiber, total dietary": 0,
            "Vitamin C, total ascorbic acid": 0
        }

        for nutrient in food['foodNutrients']:
            nutrient_name = nutrient['nutrientName']
            nutrient_value = nutrient['value']

            if nutrient_name in nutrients_totals:
                nutrients_totals[nutrient_name] += nutrient_value

        return nutrients_totals
    else:
        return None


label_to_food = {
    0: "apple",
    1: "banana",
    2: "beetroot",
    3: "bell pepper",
    4: "cabbage",
    5: "capsicum",
    6: "carrot",
    7: "cauliflower",
    8: "chilli pepper",
    9: "corn",
    10: "cucumber",
    11: "eggplant",
    12: "garlic",
    13: "ginger",
    14: "grapes",
    15: "jalapeno",
    16: "kiwi",
    17: "lemon",
    18: "lettuce",
    19: "mango",
    20: "onion",
    21: "orange",
    22: "paprika",
    23: "pear",
    24: "peas",
    25: "pineapple",
    26: "pomegranate",
    27: "potato",
    28: "radish",
    29: "soy beans",
    30: "spinach",
    31: "sweetcorn",
    32: "sweet potato",
    33: "tomato",
    34: "turnip",
    35: "watermelon",
}

st.title("Fruit and Vegetable Classifier")
st.write("Upload an image of a fruit or vegetable for classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    model = load_model()
    label = classify_image(model, image)

    food_name = label_to_food.get(label, "Unknown Food")

    if food_name != "Unknown Food":
        nutrition_info = get_nutrition_info(food_name)

        if nutrition_info:
            st.write(f"Predicted class label: {food_name.capitalize()}")
            st.write("Nutritional information:")
            for nutrient_name, value in nutrition_info.items():
                st.write(f"{nutrient_name}: {value:.2f}")
        else:
            st.write("Nutritional information not found.")
    else:
        st.write("Label not recognized.")
