import os
import gc
import csv
import dotenv
import discord
from discord.ext import commands


# Load the token from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
dotenv.load_dotenv(dotenv_path)

# Initialize the Discord client
bot = commands.Bot(command_prefix='!')


@bot.event
async def on_ready():
    guild_count = 0

    for guild in bot.guilds:
        print(f"{guild.id} - {guild.name}")
        
        for channel in guild.channels:
            print(f"\t{channel.id} - {channel.name} - {channel.category}")
            if str(channel.category) == "Text Channels":
                messages = await channel.history(limit=10000).flatten()
                with open("messages.csv", "a") as f:
                    writer = csv.writer(f, delimiter="|")
                    for i in messages:
                        writer.writerow([channel.name, i.author, i.content.lower()])
                gc.collect()


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    await bot.process_commands(message)  # so commands will still get called
# end def on_message


if __name__ == "__main__":
    with open("messages.csv", "w") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["channel", "author", "message"])
    bot.run(os.environ["TOKEN"])

