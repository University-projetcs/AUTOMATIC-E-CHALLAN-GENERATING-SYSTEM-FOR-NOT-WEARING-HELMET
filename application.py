from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilenames
import os, glob
 

class Helmet_Detection:
    def __init__(self,root):
        self.root = root
        self.root.title("Helmet Detection")
        self.root.geometry("800x564+0+0")
        
        img=Image.open('./bg.jpg').resize((800,520))
        img=ImageTk.PhotoImage(img)
        label=Label(self.root,image=img)
        label.image=img
        label.place(x=0,y=64)
        
        
        label = Label(self.root, text="Helmet Detection", font=("times new roman", 30, "bold"), bg="black", fg="white",bd=4, relief=RIDGE)
        label.pack(side=TOP, fill=X)
        
    
        btn = Button(self.root, text="Upload", font=("times new roman", 20, "bold"), bg="black", fg="white",bd=4, relief=RIDGE,command=self.upload)
        btn.place(x=100, y=100)
        
        btn = Button(self.root, text="Generate Challan", font=("times new roman", 20, "bold"), bg="black", fg="white",bd=4, relief=RIDGE,command=self.generate_challan)
        btn.place(x=100, y=200)
        
        df=Frame(self.root,bd=4,relief=RIDGE,bg="white")
        df.place(x=80,y=300,width=700,height=200)
        
        self.ChallanTable=ttk.Treeview(df,columns=("challan_no","date","bike_no","reason","amount","due_date"))
        self.ChallanTable.heading("challan_no",text="Challan No.")
        self.ChallanTable.heading("date",text="Date")
        self.ChallanTable.heading("bike_no",text="Bike No.")
        self.ChallanTable.heading("reason",text="Reason")
        self.ChallanTable.heading("amount",text="Amount")
        self.ChallanTable.heading("due_date",text="Due Date")
        
        self.ChallanTable["show"]="headings"
        self.ChallanTable.column("challan_no",width=100)
        self.ChallanTable.column("date",width=100)
        self.ChallanTable.column("bike_no",width=100)
        self.ChallanTable.column("reason",width=100)
        self.ChallanTable.column("amount",width=100)
        self.ChallanTable.column("due_date",width=100)
        
        self.ChallanTable.pack(fill=BOTH,expand=1)
    
    
        
    def upload(self):
        p=askopenfilenames()
        for file in p:
            im1=Image.open(file)
            im1=im1.save("./upload/"+file.split("/")[-1]) #save image in the same folder
    
    def generate_challan(self):
        from challan import challan
        if len(challan)==0:
            for record in self.ChallanTable.get_children():
                self.ChallanTable.delete(record)
        if len(challan)!=0:
            for record in self.ChallanTable.get_children():
                self.ChallanTable.delete(record)
            for row in challan:
                self.ChallanTable.insert('',END,values=row)
        
        files = glob.glob('./upload/*')
        for f in files:
            os.remove(f)
    
        
if __name__ == "__main__":
    root = Tk()
    obj = Helmet_Detection(root)
    root.mainloop()
    
