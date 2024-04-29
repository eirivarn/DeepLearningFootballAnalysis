import numpy as np
import matplotlib.pyplot as plt
import cv2

# Define the football field coordinates with their class IDs for clarity in visualization
football_field_coordinates = np.array([
    [45.0, 565.0], 
    [457.5, 565.0],
    [45.0, 455.0], 
    [175.0, 455.0],
    [175.0, 350.0],
    [175.0, 250.0],
    [175.0, 140.0],
    [45.0, 370.0], [85.0, 370.0],
    [85.0, 230.0],
    [45, 230.0], [130.0, 300.0], [457.5, 370.0], [457.5, 300.0], [457.5, 225.0],
    [200.0, 300.0], [385.0, 300.0], [530.0, 300.0], [870.0, 565.0], [870.0, 455.0],
    [740.0, 455.0], [740.0, 350.0], [740.0, 250.0], [740.0, 140.0], [870.0, 370.0],
    [830.0, 370.0], [830.0, 230.0], [715.0, 300.0]
])
keypoint_ids = range(len(football_field_coordinates))  # Assign an ID to each keypoint

def draw_football_field(keypoints=football_field_coordinates, keypoint_ids=keypoint_ids, player_coords=None, filename='football_field.png'):
    plt.figure(figsize=(14, 7))

    # Draw field boundaries, centerline, and center circle
    plt.plot([45, 870], [565, 565], 'green')
    plt.plot([45, 870], [30, 30], 'green')
    plt.plot([45, 45], [30, 565], 'green')
    plt.plot([870, 870], [30, 565], 'green')
    plt.plot([457.5, 457.5], [30, 565], 'green')
    center_circle = plt.Circle((457.5, 297.5), 70, color='green', fill=False)
    plt.gca().add_patch(center_circle)

    # Draw goal areas
    plt.plot([45, 175], [455, 455], 'green')
    plt.plot([45, 175], [140, 140], 'green')
    plt.plot([870, 740], [455, 455], 'green')
    plt.plot([870, 740], [140, 140], 'green')
    plt.plot([175, 175], [455, 140], 'green')
    plt.plot([740, 740], [455, 140], 'green')
    plt.plot([45, 85], [370, 370], 'green')
    plt.plot([85, 85], [370, 230], 'green')
    plt.plot([85, 45], [230, 230], 'green')
    plt.plot([870, 830], [370, 370], 'green')
    plt.plot([830, 830], [370, 230], 'green')
    plt.plot([830, 870], [230, 230], 'green')

    # Draw keypoints with annotations for their class IDs
    for idx, point in enumerate(keypoints):
        plt.scatter(*point, color='blue')
        plt.text(point[0] + 10, point[1], str(keypoint_ids[idx]), color='black', fontsize=9)

    # Draw player coordinates if provided
    if player_coords is not None:
        for coord in player_coords:
            plt.scatter(coord[0], coord[1], color='red')

    plt.title("Football Field Layout with Keypoint IDs")
    plt.xlim(0, 915)
    plt.ylim(0, 595)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()