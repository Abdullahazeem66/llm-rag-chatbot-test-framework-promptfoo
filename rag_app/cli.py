import argparse
import json
import sys

from rag_app import RAGPipeline


def _print_sources(chunks):
    print("\nSources:")
    for c in chunks:
        print(f"  {c.rank}. {c.id}  (score {c.score:.3f})  {c.section}")


def cmd_ingest(args):
    stats = RAGPipeline(args.config, auto_index=False).ingest(rebuild=args.rebuild)
    status = "built" if stats.rebuilt else "up to date"
    print(f"Index {stats.collection} {status}: {stats.documents} documents, {stats.chunks} chunks, "
          f"{stats.embedding_tokens} embedding tokens")


def cmd_retrieve(args):
    resp = RAGPipeline(args.config).retrieve(args.query, args.top_k)
    if args.json:
        print(resp.model_dump_json(indent=2))
        return
    for c in resp.retrieved:
        print(f"{c.rank}. {c.id}  (score {c.score:.3f})\n{c.text}\n")


def cmd_ask(args):
    resp = RAGPipeline(args.config).answer(args.query, top_k=args.top_k)
    if args.json:
        print(resp.model_dump_json(indent=2))
        return
    print(resp.answer)
    _print_sources(resp.retrieved)
    print(f"\n{resp.timings_ms['total']:.0f} ms | {resp.usage.input_tokens}+{resp.usage.output_tokens} tokens"
          f" | ${resp.usage.cost_usd}")


def cmd_chat(args):
    pipeline = RAGPipeline(args.config)
    history: list[dict] = []
    print("Bree - Brindlework support assistant. Type 'exit' to quit.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        resp = pipeline.answer(query, history=history)
        print(f"\nOrbi: {resp.answer}")
        if args.sources:
            _print_sources(resp.retrieved)
        print()
        history += [{"role": "user", "content": query}, {"role": "assistant", "content": resp.answer}]
        history = history[-2 * args.max_turns:]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rag_app", description="Brindlework support RAG assistant")
    parser.add_argument("--config", default="baseline", help="config name in configs/ or path to YAML")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="build the vector index")
    p.add_argument("--rebuild", action="store_true")
    p.set_defaults(func=cmd_ingest)

    for name, func in (("ask", cmd_ask), ("retrieve", cmd_retrieve)):
        p = sub.add_parser(name)
        p.add_argument("query")
        p.add_argument("--top-k", type=int)
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=func)

    p = sub.add_parser("chat", help="interactive chat")
    p.add_argument("--sources", action="store_true")
    p.add_argument("--max-turns", type=int, default=5)
    p.set_defaults(func=cmd_chat)

    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args.func(args)


if __name__ == "__main__":
    main()
