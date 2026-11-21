# gograph

Indexador estático e visualizador de grafo de chamadas para projetos **Go**, armazenado em SQLite (`.gograph/graph.db`).

O `gograph` analisa pacotes Go com `golang.org/x/tools/go/packages`, constrói o grafo de chamadas via SSA/VTA (ou CHA), extrai assinaturas, tipos e hashes de corpo, e expõe comandos para indexação contínua, auditoria de contexto, visualização web e avaliação automatizada de implementação de funções.

## Requisitos

- Go `1.27.0` (ou compatível com `go.mod`)
- SQLite (via `github.com/mattn/go-sqlite3`)

## Instalação / Compilação

```bash
go build -o bin/gograph ./cmd/gograph
```

## Comandos principais

### 1. Indexar o projeto (`index`)

Gera ou atualiza `.gograph/graph.db` com todas as funções, assinaturas, posições de arquivo e arestas de chamada (`static`, `interface`, `go`, `defer`, `closure`, `dynamic`).

```bash
gograph index .
# Incluindo arquivos _test.go:
gograph index -tests .
# Modo rápido apenas com CHA (sem VTA):
gograph index -cha .
```

### 2. Monitoramento contínuo (`watch`)

Monitora alterações em arquivos `.go` e `go.mod` no projeto e reindexa o grafo automaticamente sempre que o código muda, mantendo `.gograph/graph.db` sincronizado em tempo real.

```bash
gograph watch .
# Com testes incluídos e intervalo customizado:
gograph watch -tests -interval 1s .
```

### 3. Visualizador web do grafo (`server`)

Sobe um servidor HTTP local com interface **React + Tailwind CSS** embutida para explorar pacotes, filtrar funções, navegar pelo grafo de chamadas (`calls` e `calledBy`) e visualizar o código-fonte de cada função mapeada.

```bash
gograph server .
# Porta customizada ou banco específico:
gograph server -addr :8080 -db .gograph/graph.db .
```

Acesse `http://localhost:8080` no navegador.

### 4. Descrições e contexto (`describe` e `audit`)

```bash
# Gera descrições estruturadas das funções pendentes no banco:
gograph describe .

# Audita a cobertura do contexto gerado para funções do projeto:
gograph audit -ctx real -budget 24000 -n 50 .
```

### 5. Benchmark de implementação (`bench`)

Mede compilação (`goimports` + `go build` + `go vet` via `-overlay`, sem alterar arquivos do projeto) e execução dos testes unitários que cobrem cada função:

```bash
gograph bench -ctx real -budget 24000 -n 60 -tests -workers 12 .
```

## Estrutura do banco (`.gograph/graph.db`)

- `functions`: `id`, `name`, `package`, `file`, `line_start`, `line_end`, `signature`, `hash`, `external`, `description`
- `edges`: `caller_id`, `callee_id`, `kind`
- `meta`: metadados do módulo (`module`, `root`, `last_commit`)
