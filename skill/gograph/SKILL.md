---
name: gograph
description: Navegação, indexação contínua e implementação guiada pelo grafo de funções Go com gograph (.gograph/graph.db).
---

# Skill: gograph

Use o `gograph` como fonte principal para entender a arquitetura, dependências, tipos e fluxo de chamadas de qualquer projeto Go antes de propor ou escrever código.

## Regra central: sempre guiado pelo grafo

1. **Mantenha o grafo atualizado com `watch`**:
   - Ao trabalhar em um projeto Go, prefira manter `gograph watch .` em execução em segundo plano.
   - Sempre que houver mudanças no código-fonte (`.go` ou `go.mod`), o modo `watch` reindexa o projeto e atualiza `.gograph/graph.db` automaticamente.
   - Caso o `watch` não esteja ativo, rode `gograph index .` antes de consultar o grafo e após editar arquivos.

2. **Consulte o grafo antes de ler ou editar arquivos manualmente**:
   - Verifique em `.gograph/graph.db` (ou via `gograph server`) os nós (`functions`) e arestas (`edges`) do pacote alvo:
     - **Quem chama a função** (`calledBy`) e **o que ela chama** (`calls`).
     - **Assinaturas exatas**, tipos, campos e pacotes importados.
     - **Descrições** (`description`) registradas no banco para entender o papel de cada função sem precisar ler o repositório inteiro.

3. **Fluxo recomendado de desenvolvimento**:
   1. Inicie `gograph watch .` (ou `gograph index .`) na raiz do módulo Go.
   2. Opcionalmente, suba `gograph server -addr :8080 .` para inspeção visual do grafo de chamadas e do código de cada função.
   3. Use as informações do grafo (assinaturas, chamadores, chamados e tipos do pacote) para guiar cada alteração ou nova função.
   4. Após modificar o código, aguarde a atualização automática do `watch` (ou confirme com `go test ./...`) antes da próxima etapa.

## Comandos úteis

```bash
# Indexação única
gograph index .

# Atualização contínua do grafo a cada alteração no código
gograph watch .

# Interface visual (React + Tailwind) para navegar no grafo e ver o código-fonte
gograph server -addr :8080 .
```
