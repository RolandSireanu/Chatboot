import asyncio
import json
import os.path
import textwrap

from websockets.asyncio.client import connect

async def sendTextMessage(websocket, message):
    jsonText = json.dumps({"type": "text", "data": message})
    await websocket.send(jsonText)
    msg = await websocket.recv()
    print(textwrap.fill(msg, width=80))

async def sendData(websocket, filePath:str):
    with open(filePath, "rb") as f:
        stepSize = 1024*1024
        content = f.read()
        jsonData = json.dumps({"type": "file", "filename":os.path.basename(filePath), "size": len(content)})
        print("Sending file")
        await websocket.send(jsonData)
        for i in range(0, len(content), stepSize):
            await websocket.send(content[i : min(i+stepSize, len(content))])

# upload: /Users/sireanuroland/Chatboot/app/resources/Hacking.pdf
# upload: /Users/sireanuroland/Chatboot/app/resources/TacticalBook.pdf

async def hello():
    async with connect("ws://localhost:8000/ws") as websocket:
        inText = ""
        while(True):
            inText = await asyncio.to_thread(input, "Enter a message to send to the server or type 'exit' to quit: ")
            inText = inText.strip()
            if(inText.lower() == "exit"):
                break

            if not inText.lower().startswith("upload:"):
                await sendTextMessage(websocket, inText)
            else:
                await sendData(websocket, inText.split(":")[1].strip())
        print("Closing connection...")



if __name__ == "__main__":
    asyncio.run(hello())
