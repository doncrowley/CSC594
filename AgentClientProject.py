import socket

HEADER = 64
PORT = 5050
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "disconnect"
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)



def rec(msg):
    received = str(client.recv(2048).decode(FORMAT))
    print(received)
def send(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    #here we buffer the string sent with blank spaces to be the correct size of the header
    send_length += b' ' * (HEADER - len(send_length))
    #sending length and message
    client.send(send_length)
    client.send(message)




A = ""


count = 0

#this will just give us the opening dialogue with the user
while A != "disconnect":
    send("testing")
    received = client.recv(2048).decode(FORMAT)
    print(received)
    if "?" in received:
        msg = input("\n")
        send(msg)
