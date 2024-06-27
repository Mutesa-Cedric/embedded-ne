import cv2
import os

def display_images_in_cluster(cluster_path):
    """
    Display all images in the specified cluster directory.
    """
    # Check if the cluster directory exists
    if not os.path.exists(cluster_path):
        print(f"The directory {cluster_path} does not exist.")
        return

    # Get all image files in the directory
    image_files = [os.path.join(cluster_path, img) for img in os.listdir(cluster_path) if img.endswith('.jpg')]

    # Display each image
    for img_file in image_files:
        img = cv2.imread(img_file)
        cv2.imshow('Cluster Image', img)
        cv2.waitKey(0)  # Wait for a key press to show the next image

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ask the user to input the cluster number
    cluster_number = input("Enter the cluster number you want to review: ")
    cluster_path = f"dataset-clusters/Cluster-{cluster_number}"

    # Display images in the selected cluster
    display_images_in_cluster(cluster_path)