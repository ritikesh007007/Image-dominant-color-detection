import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def find_dominant_color(image_path, k=3):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Error: Image not found!")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_
    labels = kmeans.labels_

    unique_labels, counts = np.unique(labels, return_counts=True)
    dominant_color_index = np.argmax(counts)
    dominant_color = colors[dominant_color_index].astype(int)

    print(f"🎨 Dominant Color (RGB): {dominant_color}")

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

   
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Original Image')

    
    ax2.pie(counts, labels=[f'Color {i+1}' for i in range(k)],
            colors=np.array(colors/255), autopct='%1.1f%%')
    ax2.set_title("Color Distribution")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    find_dominant_color('./fl.jpg', k=3)
