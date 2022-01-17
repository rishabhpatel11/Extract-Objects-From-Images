import io
import os
import PySimpleGUI as sg
from PIL import Image
from mask_rcnn import mask_rcnn
import numpy as np

file_types = [("JPEG (*.jpg)", "*.jpg"),("All files (*.*)", "*.*")]

def main():
    image1 = None
    masks = None
    boxes = None
    fig1 = None
    fig2 = None
    x_offset = None
    y_offset = None
    dragging = False
    start = None
    end = None
    statement = ""
    col1 = [
        [
            # Graph is a 1260x600 region where the images/copied objects will be created
            sg.Graph(
            canvas_size=(1260, 600),
            graph_bottom_left=(0, 600),
            graph_top_right=(1260,0),
            key="-GRAPH-",
            change_submits=True,
            background_color='white',
            drag_submits=True)
        ],
        [
            # File browser for input images
            sg.Text("Image File 1"),
            sg.Input(size=(25, 1), key="-FILE1-"),
            sg.FileBrowse(file_types=file_types),
            # Buttons to load images and identify objects in images
            sg.Button("Load Image 1"),
            sg.Button("Identify Objects"),
            sg.Text("Instructions:\
                \nLoad 2 images. You can only copy objects from the first image\
                \nClick identify objects\
                \nSelect a mode to perform an action\
                \nTip: for overlapping objects, click close to the top left of the bounding box of your desired object")
        ],
        [
            sg.Text("Image File 2"),
            sg.Input(size=(25, 1), key="-FILE2-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Load Image 2"),
        ],
        # Radio buttons to select mode
        # Delete : delete user duplicated objects
        # Move : move user duplicated objects
        # Duplicate : copy an object from image 1
        [
            sg.R('Delete new objects : Click on a copied object to delete it', 1, key='-DELETE-', enable_events=True)
        ],
        [
            sg.R('Move : Click and drag a labeled object to copy it', 1, key='-MOVE-', enable_events=True)
        ],
        [
            sg.R('Duplicate (Default) : Click on an object to create a copy', 1, True, key='-COPY-', enable_events=True)
        ],
    ]
    layout = [
        [sg.Col(col1)],
        [sg.Text(key='info', size=(60, 1))]
        ]

    # Create window and a graph
    window = sg.Window("Extract and Merge Images", layout, finalize=True)
    graph = window["-GRAPH-"]
    graph.bind('<Button-3>', '+RIGHT+')

    # Looping to check for user actions (events)
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        # Location of event(mouse)
        x, y = values["-GRAPH-"]
        # User clicked load image 1
        if event == "Load Image 1":
            filename = values["-FILE1-"]
            if os.path.exists(filename):
                # Open image, scale down to a max of 600x600(preserve aspect ratio)
                image = Image.open(values["-FILE1-"])
                image1 = image
                image.thumbnail((600, 600))
                x_offset = image.size[0]
                y_offset = image.size[1]
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                # Delete previous image if any
                if fig1 != None:
                    graph.delete_figure(fig1)
                # Draw new image on graph
                # x margin is 20
                # y margin depends on the size of the image such that margin above and below are equal
                fig1 = graph.draw_image(data=bio.getvalue(), location = (20 + (600-x_offset)/2,(600-y_offset)/2))
        # User clicked identify objects 
        if event == "Identify Objects":
            image = image1
            print(image.size)
            # Input the image into the model to get the segmented image
            image, masks, boxes, labels = mask_rcnn(image)
            print(image.shape)
            print(masks[2].shape)
            image = Image.fromarray(image)
            # Resize to max 600x600
            image.thumbnail((600, 600))
            print(image.size)
            x_offset = image.size[0]
            y_offset = image.size[1]
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            #window["-IMAGE1-"].update(data=bio.getvalue())
            # Delete previous image
            if fig1 != None:
                graph.delete_figure(fig1)
            # Draw segmented image
            fig1 = graph.draw_image(data=bio.getvalue(), location = (20 + (600-x_offset)/2,(600-y_offset)/2))
        # User clicked load image 2
        if event == "Load Image 2":
            filename = values["-FILE2-"]
            if os.path.exists(filename):
                # Open image, scale down to a max of 600x600(preserve aspect ratio)
                image = Image.open(values["-FILE2-"])
                image.thumbnail((600, 600))
                print(image.size)
                x_offset = image.size[0]
                y_offset = image.size[1]
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                #window["-IMAGE1-"].update(data=bio.getvalue())
                # Delete previous image if any
                if fig2 != None:
                    graph.delete_figure(fig2)
                # x margin is 640 i.e. end of image 1 + 20 to create a 20 px gap between images
                # y margin depends on the size of the image such that margin above and below are equal
                fig2 = graph.draw_image(data=bio.getvalue(), location = (640+(600-x_offset)/2,(600-y_offset)/2))
        # User clicked somewhere in the graph
        if event == "-GRAPH-":
            # Location of click
            x, y = values["-GRAPH-"]
            # Mode is copy
            if values['-COPY-']:
                if boxes != None:
                    # Location must be within the bounds of image 1
                    if x > 20 + (600-x_offset)/2 and x < 20 + (600-x_offset)/2 + x_offset:
                        if y > (600-y_offset)/2 and y < (600-y_offset)/2 + y_offset:
                            # Get the closest object using its bounding box
                            # Distance is measured from top left of the box to the mouse location
                            closest_box = None
                            min_dist = 10000
                            for box in boxes:
                                dist = abs((x - 20 - box[0][0])) + abs((y - (600-y_offset)/2 - box[0][1]))
                                if dist < min_dist:
                                    closest_box = box
                                    min_dist = dist
                            # Create a cropped image of the selected object
                            region = image1.crop((closest_box[0][0], closest_box[0][1], closest_box[1][0], closest_box[1][1]))
                            region.thumbnail((200, 200))
                            bio = io.BytesIO()
                            region.save(bio, format="PNG")
                            # Drop that cropped image at the top left of the graph
                            graph.draw_image(data=bio.getvalue(), location = (0,0))
                            statement = f"Duplicated object at ({x},{y})"
                else:
                    statement = "Click identify objects before trying to duplicate"
            # Mode is move
            if values['-MOVE-']:
                # Initial click
                if not dragging:
                    start = (x, y)
                    dragging = True
                    drag_figures = graph.get_figures_at_location((x,y))
                    lastxy = x, y
                else:
                    # User still has the mouse held down, keep updating mouse location
                    end = (x, y)
                # Move object
                if None not in (start, end):
                    for fig in drag_figures:
                        # Dont move intial uploaded images
                        if(fig not in [fig1, fig2]):
                            # Move object to last mouse location
                            graph.move_figure(fig, x - lastxy[0], y - lastxy[1])
                            graph.update()
                lastxy = x,y
                statement = f"Moved object from {start} tp {end}"
            # Mode is delete
            if values['-DELETE-']:
                # Get figures at the mouse location
                figures = graph.get_figures_at_location((x,y))
                # Delete all figures(user duplicated objects) expect the initial images
                for fig in figures:
                    if(fig not in [fig1, fig2]):
                        graph.delete_figure(fig)
                statement = f"Deleted object at ({x}, {y})"
        elif event.endswith('+UP'):
            # Mouse clicked
            info = window["info"]
            # Show user what action was performed in the bottom left of the window
            info.update(value="Latest activity: " + statement)
            # Reset variables for another user action
            start = None
            end = None
            dragging = False
        
    window.close()
if __name__ == "__main__":
    main()