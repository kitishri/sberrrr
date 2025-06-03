from ml.storage import upload_model, download_model

if __name__ == "__main__":
    upload_model("model_20250516.pkl", "prod_model.pkl")

    download_model("prod_model.pkl", "downloaded_model.pkl")
