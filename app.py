import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import tempfile
import os

# Load and prepare data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
labels = iris.target_names

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=labels)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = clf.score(X_test, y_test)
model_summary = f"Model: RandomForestClassifier\nTraining size: {len(X_train)}\nTest size: {len(X_test)}\nAccuracy: {accuracy:.2f}"

# Prediction
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=X.columns)
    prediction = clf.predict(input_df)[0]
    return f"🌸 Prediction: {labels[prediction]}"

# Confusion Matrix Plot
def plot_confusion_matrix():
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="YlGnBu", ax=ax)
    fig_path = os.path.join(tempfile.gettempdir(), "confusion_matrix.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    return fig_path

# Feature Importance Plot
def plot_feature_importance():
    fig, ax = plt.subplots()
    pd.Series(clf.feature_importances_, index=X.columns).sort_values().plot(kind='barh', color='teal', ax=ax)
    fig_path = os.path.join(tempfile.gettempdir(), "feature_importance.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    return fig_path

# PDF Generation
def generate_pdf():
    conf_path = plot_confusion_matrix()
    feat_path = plot_feature_importance()
    pdf_path = os.path.join(tempfile.gettempdir(), "iris_report.pdf")

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Summary & Report
    text = c.beginText(40, height - 40)
    text.setFont("Helvetica-Bold", 12)
    text.textLine("Model Summary")
    text.setFont("Helvetica", 10)
    for line in model_summary.split("\n"):
        text.textLine(line)
    c.drawText(text)

    text = c.beginText(40, height - 180)
    text.setFont("Helvetica-Bold", 12)
    text.textLine("Classification Report")
    text.setFont("Helvetica", 10)
    for line in report.split("\n"):
        text.textLine(line)
    c.drawText(text)

    c.showPage()

    # Plots
    c.drawImage(ImageReader(conf_path), 40, height - 470, width=250, preserveAspectRatio=True)
    c.drawImage(ImageReader(feat_path), 320, height - 470, width=250, preserveAspectRatio=True)

    c.showPage()
    c.save()

    return pdf_path

# Gradio UI
with gr.Blocks(title="🌸 Iris Classifier") as demo:
    gr.Markdown("# 🌸 Iris Classifier - Gradio App")
    gr.Markdown(f"**Model Accuracy:** {accuracy:.2f}")

    with gr.Tab("🔮 Predict"):
        with gr.Row():
            sepal_length = gr.Slider(minimum=X.iloc[:, 0].min(), maximum=X.iloc[:, 0].max(), value=X.iloc[:, 0].mean(), label="Sepal Length (cm)")
            sepal_width = gr.Slider(minimum=X.iloc[:, 1].min(), maximum=X.iloc[:, 1].max(), value=X.iloc[:, 1].mean(), label="Sepal Width (cm)")
            petal_length = gr.Slider(minimum=X.iloc[:, 2].min(), maximum=X.iloc[:, 2].max(), value=X.iloc[:, 2].mean(), label="Petal Length (cm)")
            petal_width = gr.Slider(minimum=X.iloc[:, 3].min(), maximum=X.iloc[:, 3].max(), value=X.iloc[:, 3].mean(), label="Petal Width (cm)")
        output = gr.Textbox(label="Prediction")
        predict_btn = gr.Button("Predict")
        predict_btn.click(predict_iris, inputs=[sepal_length, sepal_width, petal_length, petal_width], outputs=output)

    with gr.Tab("📄 Classification Report"):
        gr.Textbox(value=report, label="Classification Report", lines=20)

    with gr.Tab("📊 Confusion Matrix"):
        matrix_img = gr.Image(label="Confusion Matrix")
        matrix_btn = gr.Button("Generate Plot")
        matrix_btn.click(fn=plot_confusion_matrix, inputs=[], outputs=matrix_img)

    with gr.Tab("📌 Feature Importance"):
        feat_img = gr.Image(label="Feature Importance")
        feat_btn = gr.Button("Generate Plot")
        feat_btn.click(fn=plot_feature_importance, inputs=[], outputs=feat_img)

    with gr.Tab("📥 Download PDF Report"):
        pdf_output = gr.File(label="PDF Report")
        pdf_btn = gr.Button("Generate PDF")
        pdf_btn.click(fn=generate_pdf, inputs=[], outputs=pdf_output)

# demo.launch()
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
