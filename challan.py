from main import plates
import mysql.connector as msql
from mysql.connector import Error
import smtplib
import ssl
from email.message import EmailMessage
import datetime
import random


email=[]

try:
    conn = msql.connect(host='localhost', database='dummy1', user='root', password='ranga12345')
    if conn.is_connected():
        cursor = conn.cursor()
        for x in plates:
            sql = "SELECT email FROM dummy1.dummydata where plate = '"+x+"' "
            cursor.execute(sql)
            result = cursor.fetchall()
            if len(result) == 0:
                result = [""]
            for i in result:
                email.append(i)
except Error as e:
    print("Error while connnecting to MySQL",e)
    
print(email)


challan = []
for i in email:
    if i=="":
        continue
    challan_no=str(random.randint(1000000000,9999999999))
    date=str(datetime.datetime.today())
    bikeno=str(plates[email.index(i)])
    duedate=str((datetime.datetime.today() + datetime.timedelta(days=30))) 
    
    #message
    subject = 'E-challan copy from GCPD Police'
    body="""
    This is an online copy of e-challan for your vechile number '"""+str(plates[email.index(i)])+"""'

    Challan No. : """ + challan_no +"""
    Date        : """ + date + """
    Bike No.    : """ + bikeno + """
    Reason      : HELMET VOILATION
    Amount      : ₹ 150.00
    Due Date    : """ + duedate + """

    E-payment link - https://echallan.parivahan.gov.in/index/accused-challan  
    

    :::HURT YOUR HELMET NOT YOUR HEAD:::
    """
    
    
    email_sender = 'srirangachakilam@gmail.com'
    email_password = 'jziokuopcpfgvpsu'
    email_receiver = i
    
    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = i
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, i, em.as_string())

        print('\nsent mail to', i, 'for vechile number', plates[email.index(i)])
    k = (challan_no, date, bikeno,"HELMET VIOLATION","150/-", duedate)
    
    
    challan.append(k)
    
        
