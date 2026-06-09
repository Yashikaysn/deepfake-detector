from facenet_pytorch import MTCNN
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import os

# This automatically finds and crops faces from photos
face_finder = MTCNN(margin=20)

def crop_face(image_path, save_path):
    image = Image.open(image_path)
    face = face_finder(image)

    if face is not None:
        # face is a tensor, convert back to image
        face_image = to_pil_image((face * 0.5 + 0.5).clamp(0, 1))
        face_image.save(save_path)
    else:
        print(f"No face found in: {image_path}")

def crop_all_faces(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    files = os.listdir(input_folder)

    for i, filename in enumerate(files):
        if filename.endswith(".jpg"):
            crop_face(
                f"{input_folder}/{filename}",
                f"{output_folder}/{filename}"
            )
            print(f"Processed {i+1}/{len(files)} — {filename}")

    print("All faces cropped!")