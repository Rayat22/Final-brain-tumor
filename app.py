import streamlit as st
import torch
from torch import nn
from torchvision import transforms # type: ignore
from PIL import Image
import os


# Step 1: Load the trained model
MODEL_PATH = r'C:\Users\r4y4t\Desktop\Brain tumor detector\All\brain_tumor_model.pth'
# Define your model architecture
class ConvNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(12, 20, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(20, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(75 * 75 * 32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = x.view(-1, 32 * 75 * 75)
        x = self.fc(x)
        return x

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet(num_classes=4).to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
else:
    raise FileNotFoundError(f"The model file was not found at {MODEL_PATH}")

# Step 2: Create the login page
def login_page():
    st.title("Brain Tumor Detection")
    st.subheader("Please enter your username and password to log in.")
    st.markdown(
    
    """
    <style>
    .stApp {
        background-image: url('https://github.com/Rayat22/Final-brain-tumor/blob/main/Background.png?raw=true');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center center;
    }

    .login-container {
        background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent background */
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        width: 50%;
        margin: auto;
    }

    .login-container h1, .login-container h2, .login-container label, .login-container button {
        color: #333; /* Dark text for readability */
    }

    .login-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        font-size: 16px;
        cursor: pointer;
        border-radius: 5px;
    }
    .login-button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)



    # Create login form inside a styled container
    with st.form(key='login_form'):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Log In", use_container_width=True)
        
        if submit_button:
            if username == "admin" and password == "admin":  # Secure this in production
                st.session_state.logged_in = True
                st.success("Login successful!")
                return True
            else:
                st.error("Invalid credentials. Please try again.")
                return False
    return False

# Step 3: Create image upload and prediction function
def predict_tumor_type(image):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
    
    _, predicted = torch.max(output, 1)
    classes = ['glioma_tumor', 'meningioma_tumor','no_tumor', 'pituitary_tumor']
    predicted_class = classes[predicted.item()]
    return predicted_class

# Step 4: Main content after successful login
def main_page():
    st.title("Upload an MRI Image for Tumor Detection")

    # Upload image
    uploaded_image = st.file_uploader("Choose an MRI image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Prediction
        st.text("Classifying the tumor:")
        result = predict_tumor_type(image)
        st.markdown(f"<h3 style='color: #333; font-weight: bold;'>Predicted tumor type: {result}</h3>", unsafe_allow_html=True)

# Step 5: App flow
def app():
    if "logged_in" not in st.session_state:
        if login_page():
            main_page()
    else:
        main_page()

if __name__ == "__main__":
    app()
