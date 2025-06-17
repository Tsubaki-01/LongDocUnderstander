import base64


#  base 64 编码格式
def encode_image_base_64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
