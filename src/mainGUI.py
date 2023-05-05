from tkinter import *
from tkinter import filedialog as fd


root = Tk()  # create root window
root.title("Basic GUI Layout with Grid")
root.maxsize(900, 600)  # width x height
root.config(bg="skyblue")

# Create left and right frames
left_frame = Frame(root, width=200, height=400, bg="grey")
left_frame.grid(row=0, column=0, padx=10, pady=5)

right_frame = Frame(root, width=650, height=400, bg="grey")
right_frame.grid(row=0, column=1, padx=10, pady=5)

left_frame.grid_propagate(False)
right_frame.grid_propagate(False)

error_Frame = Frame(root, width=400, height=30, bg="red")
error_Frame.grid(
    row=1, column=0, columnspan=2, padx=10, pady=5, sticky="s"
)  # Set row to 1 and columnspan to 2

error_Frame.grid_propagate(False)


tool_bar = Frame(left_frame, width=180, height=185, bg="grey")
tool_bar.grid(row=2, column=0, padx=5, pady=5)


mainLabel = Label(tool_bar, text="Tools", relief=RAISED)
mainLabel.grid(row=0, column=0, padx=5, pady=3, ipadx=10)

ownerLabel = Label(right_frame, text="Owner", relief=RAISED)
ownerLabel.grid(row=0, column=0, padx=5, pady=5, ipadx=10)

errorLabel = Label(error_Frame, text="Err", relief=RAISED)
errorLabel.grid(
    row=0, column=0, padx=5, pady=3, ipadx=10, sticky="nsew", in_=error_Frame
)
error_Frame.grid_rowconfigure(0, weight=1)
error_Frame.grid_columnconfigure(0, weight=1)


def handleIdentify():
    try:
        print("Owner of the file" + fileText + " is " + "Tom")
        labelText = "Owner of the file: " + fileText + " is " + '"Tom"'
        ownerLabel.config(text=labelText)
    except:
        print("Error occured")
        errorLabel.config(text="Error occured")


def handleSelect():
    fileSelector = fd.askopenfilename()
    #   print("File selector is", fileSelector)
    global fileText
    fileText = getName(fileSelector)

    if mainLabel:
        mainLabel.config(text=fileText)


def getName(dir):
    dir = dir[::-1]
    retVal = ""
    for c in range(0, len(dir)):
        if dir[c] != "/":
            retVal += dir[c]

        else:
            break

    return retVal[::-1]


Button(tool_bar, text="Select Sound", command=handleSelect).grid(
    row=1, column=0, padx=5, pady=5, sticky="w" + "e" + "n" + "s"
)
Button(tool_bar, text="Identify", command=handleIdentify).grid(
    row=2, column=0, padx=5, pady=5, sticky="w" + "e" + "n" + "s"
)
# Button(tool_bar,  text="Rotate &amp; Flip",  command=handleRotate).grid(row=3,  column=0,  padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
# Button(tool_bar,  text="Resize",  command=handleResize).grid(row=4,  column=0,  padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
# Button(tool_bar,  text="Black &amp; White",  command=handleBW).grid(row=1,  column=1,  padx=5,  pady=5,  sticky='w'+'e'+'n'+'s')
root.mainloop()
