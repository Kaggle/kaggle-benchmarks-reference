src/
  main.rs                  Entry point: index, compute, start server
  lib.rs                   Library root (re-exports)
  cli.rs                   CLI argument parsing (19 subcommands)
  cli_runner.rs            CLI subcommand dispatcher
  config.rs                Project configuration and root detection
  error.rs                 Error types
  str_utils.rs             String utilities (stable floor_char_boundary polyfill)
  toolchain.rs             Toolchain detection (Cargo, npm, Go, etc.)
  watch.rs                 File watcher for incremental re-indexing
  guard.rs                 Modification guard evaluation engine
  embeddings.rs            Local embedding model for qartez_semantic (opt-in)
  server/
    mod.rs                 MCP server entrypoint - dispatches to per-tool handlers
    tools/                 30 per-tool handler modules (one file per MCP tool)
    prompts.rs             6 workflow prompt templates
    tiers.rs               Progressive tool disclosure (core/analysis/refactor/meta)
    cache.rs               Tree-sitter parse cache
    helpers.rs             Shared handler utilities
    overview.rs            Overview/map generation
    params.rs              Tool parameter structs
    treesitter.rs          Tree-sitter integration helpers
    mcp_instructions.md    Embedded MCP server instructions
  index/
    mod.rs                 Core indexing engine (full + incremental, import resolution)
    walker.rs              File discovery (respects .gitignore + .qartezignore)
    parser.rs              Tree-sitter parser pool
    symbols.rs             Symbols / imports / references + AST shape hashing
    languages/             37 language adapters (21 with cyclomatic complexity)
  graph/
    mod.rs                 Graph module root
    pagerank.rs            PageRank on import graph
    blast.rs               Blast radius BFS
    leiden.rs              Community detection (Leiden clustering)
    boundaries.rs          Architecture-boundary rules engine
    security.rs            Security rule engine (powers qartez_security)
    wiki.rs                Architecture wiki renderer
  git/
    mod.rs                 Git module root
    cochange.rs            Co-change pair mining
    diff.rs                Diff range analysis (for qartez_diff_impact)
    trend.rs               Complexity trend over git history
    knowledge.rs           Code authorship and bus-factor analysis
  storage/
    mod.rs                 Storage module root
    schema.rs              SQLite + FTS5 schema
    read.rs / write.rs     Query and mutation helpers
    models.rs              Row structs
  bin/
    setup.rs               Interactive IDE setup wizard (19 IDEs)
    guard.rs               PreToolUse modification guard
    benchmark.rs           Benchmark harness entry point
  benchmark/               Benchmark internals (cargo feature)
    profiles/              Per-language benchmark profiles (Rust, TS, Python, Go, Java)
    scenarios.rs           28 benchmark scenarios
    judge.rs               LLM-judge harness
    report.rs              Markdown / JSON report writers
    tokenize.rs            cl100k_base token accounting
scripts/                   Hook + snippet assets embedded by qartez-setup
benchmarks/fixtures.toml   Pinned OSS repos for multi-language benchmarks
reports/               
