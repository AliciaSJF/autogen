from typing import AsyncGenerator, Sequence, List
from autogen_agentchat.agents import BaseChatAgent
# from autogen_agentchat.base import Response
# from autogen_agentchat.messages import ChatMessage, TextMessage, AgentEvent
# from autogen_core import CancellationToken

# class CountDownAgent(BaseChatAgent):
#     def __init__(self, name: str, count: int = 3):
#         super().__init__(name, "A simple agent that counts down.")
#         self._count = count  # Número inicial de la cuenta regresiva

#     @property
#     def produced_message_types(self) -> Sequence[type[ChatMessage]]:
#         return (TextMessage,)

#     async def on_messages_stream(
#         self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
#     ) -> AsyncGenerator[AgentEvent | ChatMessage | Response, None]:
#         inner_messages: List[AgentEvent | ChatMessage] = []
#         for i in range(self._count, 0, -1):
#             msg = TextMessage(content=f"{i}...", source=self.name)
#             inner_messages.append(msg)
#             yield msg  # Emitir mensaje de la cuenta regresiva
        
#         yield Response(chat_message=TextMessage(content="Done!", source=self.name), inner_messages=inner_messages)

#     async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
#         response: Response | None = None
#         async for message in self.on_messages_stream(messages, cancellation_token):
#             if isinstance(message, Response):
#                 response = message
#         return response

#     async def on_reset(self, cancellation_token: CancellationToken) -> None:
#         pass  # No necesita reiniciarse

# # Ejecutar el agente de cuenta regresiva
# import asyncio

# async def run_countdown_agent():
#     countdown_agent = CountDownAgent("countdown", count=5)
#     async for message in countdown_agent.on_messages_stream([], CancellationToken()):
#         print(message.content if isinstance(message, TextMessage) else message.chat_message.content)

# # Ejecutar en un script
# asyncio.run(run_countdown_agent())

from typing import Callable, Sequence, List
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ChatMessage, TextMessage
from autogen_core import CancellationToken

class ArithmeticAgent(BaseChatAgent):
    def __init__(self, name: str, description: str, operator_func: Callable[[int], int]) -> None:
        super().__init__(name, description=description)
        self._operator_func = operator_func
        self._message_history: List[ChatMessage] = []

    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        # Guardar el historial de mensajes
        self._message_history.extend(messages)
        # Obtener el número de la última entrada del usuario
        assert isinstance(self._message_history[-1], TextMessage)
        number = int(self._message_history[-1].content)
        # Aplicar la operación matemática
        result = self._operator_func(number)
        # Crear mensaje de respuesta
        response_message = TextMessage(content=str(result), source=self.name)
        self._message_history.append(response_message)
        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass  # No necesita reiniciarse

from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio

# Crear instancias del ArithmeticAgent con diferentes operaciones
add_agent = ArithmeticAgent("add_agent", "Suma 1 al número.", lambda x: x + 1)
multiply_agent = ArithmeticAgent("multiply_agent", "Multiplica por 2.", lambda x: x * 2)
subtract_agent = ArithmeticAgent("subtract_agent", "Resta 1.", lambda x: x - 1)
divide_agent = ArithmeticAgent("divide_agent", "Divide entre 2 (redondea hacia abajo).", lambda x: x // 2)
identity_agent = ArithmeticAgent("identity_agent", "Devuelve el mismo número.", lambda x: x)

# Condición de terminación: máximo 10 mensajes
termination_condition = MaxMessageTermination(10)

# Crear SelectorGroupChat con los agentes matemáticos
selector_group_chat = SelectorGroupChat(
    [add_agent, multiply_agent, subtract_agent, divide_agent, identity_agent],
    model_client=OpenAIChatCompletionClient(model="gpt-4o"),
    termination_condition=termination_condition,
    allow_repeated_speaker=True,  # Permitir que el mismo agente hable varias veces
    selector_prompt=(
        "Available roles:\n{roles}\nTheir job descriptions:\n{participants}\n"
        "Current conversation history:\n{history}\n"
        "Please select the most appropriate role for the next message, and only return the role name."
    ),
)

# Ejecutar el equipo con una entrada inicial
async def run_number_agents():
    task = [
        TextMessage(content="Apply the operations to turn the given number into 25.", source="user"),
        TextMessage(content="10", source="user"),
    ]
    await Console(selector_group_chat.run_stream(task=task))

# Ejecutar en un script
asyncio.run(run_number_agents())
