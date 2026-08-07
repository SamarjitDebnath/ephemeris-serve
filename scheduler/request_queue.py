import asyncio


class RequestQueue:
    def __init__(self):
        self.queue = asyncio.Queue()

    async def put(self, request):
        await self.queue.put(request)

    async def get(self):
        return await self.queue.get()

    def empty(self) -> bool:
        return self.queue.empty()



request_queue = RequestQueue()
batch_request_queue = RequestQueue()