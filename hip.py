import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os

# Crear los agentes
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
assistant = AssistantAgent("assistant", model_client=model_client)

# Crear el equipo con un límite de 1 turno por ejecución
team = RoundRobinGroupChat([assistant], max_turns=1)

async def main():
    task = "Write a 4-line poem about the ocean."  # Inicializar task dentro de main()

    while True:
        # Ejecutar la conversación en streaming
        await Console(team.run_stream(task=task))

        # Pedir feedback al usuario
        task = input("Enter your feedback (type 'exit' to leave): ").strip()

        # Salir del bucle si el usuario lo decide
        if task.lower() == "exit":
            print("\n🚀 Terminando la conversación...")
            break

# Manejar ejecución en entornos con loops asíncronos ya en marcha
try:
    asyncio.run(main())
except RuntimeError:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
