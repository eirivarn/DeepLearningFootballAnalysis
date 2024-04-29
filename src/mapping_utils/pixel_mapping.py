import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# This list will hold the coordinates of the points clicked
points = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:  # Check if the click was inside the axes
        print(f'Pixel coordinates: x = {int(event.xdata)}, y = {int(event.ydata)}')


# Load the image
img = mpimg.imread('/Users/eirikvarnes/Downloads/footballfield.png')

# Display the image
fig, ax = plt.subplots()
imgplot = ax.imshow(img)

# Connect the click event to the onclick function
connection_id = fig.canvas.mpl_connect('button_press_event', onclick)

# Display the image
plt.show()

