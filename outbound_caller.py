import socket
import time

def originate_call(phone_number):
    s = socket.socket()
    s.connect(("127.0.0.1", 5038))  # AMI port

    # Login
    s.send(b"Action: Login\r\nUsername: ai\r\nSecret: ai_secret\r\n\r\n")
    time.sleep(0.5)

    # Send originate action
    call_str = f"""Action: Originate
Channel: PJSIP/{phone_number}
Context: ai-survey
Exten: 3001
Priority: 1
CallerID: "AI Survey" <3001>
Async: true

"""

    s.send(call_str.encode())
    time.sleep(1)  # ✅ Wait a bit before logging out

    # You can optionally read response like this:
    response = s.recv(4096).decode()
    print("📨 AMI Response:\n", response)

    # Logout
    s.send(b"Action: Logoff\r\n\r\n")
    time.sleep(0.5)
    s.close()

    print("📞 Outbound call initiated to", phone_number)

# Example usage:
originate_call("7001")
