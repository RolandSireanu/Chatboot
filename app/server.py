import asyncio
import os
import json
import shutil
import websockets
import chatboot
from websockets.asyncio.server import serve
from pathlib import Path

def GetWorkspace(workspace :str):
    workspacesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "workspaces")
    return os.path.join(workspacesPath, workspace)

class FileBuilder:
    def __init__(self, fileName: str, fileSize :int, workspacePath :str):
        self.fileName = fileName
        self.fileSize = fileSize
        self.currentSize = 0
        self.data = []
        self.workspacePath = workspacePath
    
    def addBytes(self, data: bytes) -> bool:
        self.data.append(data)
        self.currentSize += len(data)
        if self.currentSize >= self.fileSize:
            self.__writeToDisk()
            return False
        else:
            return True

    def getFilePath(self) -> str:
        return os.path.join(self.workspacePath, self.fileName)

    def __writeToDisk(self):
        Path(self.workspacePath).mkdir(parents=True, exist_ok=True)
        
        binData = b"".join(self.data)
        with open(os.path.join(self.workspacePath, self.fileName), "wb") as f:
            f.write(binData)


async def handler(websocket):
    fileReceiveState = False
    totalSize = 0
    size = 0
    fileBuilder = None
    websocketId = websocket.id.hex
    boot = chatboot.ChatBoot(GetWorkspace(websocketId))

    while True:
        try:       
            msg = await websocket.recv()
            if fileReceiveState:
                fileReceiveState = fileBuilder.addBytes(msg)
                if(not fileReceiveState):
                    boot.addBookToVectorStore(fileBuilder.getFilePath())
            else:
                jsonMsg = json.loads(msg)
                if jsonMsg["type"] == "file":
                    fileBuilder = FileBuilder(jsonMsg["filename"], jsonMsg["size"], GetWorkspace(websocketId))                    
                    fileReceiveState = True
                elif jsonMsg["type"] == "text":
                    response = boot.query(jsonMsg["data"])
                    await websocket.send(response)
                else:
                    await websocket.send("Invalid message type")
        except websockets.ConnectionClosed:
            print(f"Connection closed for {websocketId}")
            

            if(os.path.exists(GetWorkspace(websocketId))):
                shutil.rmtree(GetWorkspace(websocketId))
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            if(os.path.exists(GetWorkspace(websocketId))):
                shutil.rmtree(GetWorkspace(websocketId))
            break


async def main():
    async with websockets.serve(handler, "0.0.0.0", 8000) as server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
    workspacePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "workspaces")
    if (os.path.exists(workspacePath)):
        shutil.rmtree(workspacePath)