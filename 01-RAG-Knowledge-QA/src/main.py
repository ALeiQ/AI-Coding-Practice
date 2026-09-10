from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

app = typer.Typer(help="RAG Knowledge QA System")
console = Console()


@app.command()
def ingest(
    path: str = typer.Argument("data", help="Path to documents directory or file"),
    recreate: bool = False,
):
    """Load documents from path and index into Qdrant."""
    from src.ingest.pipeline import ingest_path

    with console.status("[bold green]Ingesting documents..."):
        result = ingest_path(path, recreate=recreate)

    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[green]Ingestion complete![/green]\n"
            f"Documents: {result['documents']}\n"
            f"Chunks: {result['chunks']}",
            title="Ingest",
        )
    )


@app.command()
def query(question: str = typer.Argument(..., help="Question to ask")):
    """Ask a question against the knowledge base."""
    from src.qa.chain import answer_question

    with console.status("[bold blue]Searching and generating answer..."):
        result = answer_question(question)

    console.print("\n[bold]Answer:[/bold]")
    console.print(Markdown(result["answer"]))

    if result["sources"]:
        console.print(f"\n[dim]Sources ({result['chunks_used']} chunks used):[/dim]")
        for src in set(result["sources"]):
            console.print(f"  - {src}")


@app.command()
def status():
    """Show collection status."""
    from src.vectorstore.store import collection_info, get_client

    client = get_client()
    info = collection_info(client)
    if info is None:
        console.print("[yellow]No collection found. Run 'rag ingest' first.[/yellow]")
        raise typer.Exit(0)

    console.print(
        Panel(
            f"Collection: {info['name']}\n"
            f"Points: {info['points_count']}\n"
            f"Status: {info['status']}",
            title="Qdrant Status",
        )
    )


if __name__ == "__main__":
    app()
