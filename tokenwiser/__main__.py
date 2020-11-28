import typer

from tokenwiser import __version__

app = typer.Typer(
    add_completion=False,
    help="Tokenwiser CLI. Allows you to train embeddings from the commandline.",
)


@app.command("version", help="show the version of tokenwise")
def version():
    typer.echo(f"{__version__}")


@app.command()
def init():
    pass


if __name__ == "__main__":
    app()
