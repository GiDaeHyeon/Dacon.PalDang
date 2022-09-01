from typing import Optional
from urllib.parse import urlencode

from retry import retry
import xmltodict
import aiohttp
from aiohttp.client_reqrep import ClientResponse
from aiohttp.client_exceptions import ContentTypeError


class AsyncCrawler:
    def __init__(self) -> None:
        self.parser = xmltodict.parse

    async def response_parser(self, response: ClientResponse, parse: str = 'json'):
        if response.status >= 300:
            raise ConnectionError(f'{response.url}이 응답하지 않습니다.(status code: {response.status})')

        if parse == 'json':
            try:
                resp = await response.json()
                return resp
            except ContentTypeError:
                resp = await response.text()
                return self.parser(resp)
            except Exception as E:
                print(E)
        elif parse == 'text':
            resp = await response.text()
            return resp

    @retry(exceptions=(ConnectionError, ), tries=5, delay=10, max_delay=60)
    async def get(self, url: str, params: Optional[dict] = None, data: Optional[dict] = None,
                  headers: Optional[dict] = None, parse: str = 'json') -> str:
        async with aiohttp.ClientSession(headers=headers) as session:
            converted_params = '?' + urlencode(params)
            async with session.get(url + converted_params, data=data) as response:
                return await self.response_parser(response=response, parse=parse)

    @retry(exceptions=(ConnectionError, ), tries=5, delay=10, max_delay=60)
    async def post(self, url: str, data: Optional[dict] = None, headers: Optional[dict] = None, parse: str = 'json'):
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(url=url, data=data) as response:
                return await self.response_parser(response=response, parse=parse)
