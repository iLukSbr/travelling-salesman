# Modelagem Computacional do Problema do Caixeiro Viajante

Este projeto apresenta a modelagem computacional do Problema do Caixeiro Viajante (TSP) utilizando **Têmpera Simulada (Simulated Annealing, SA)** e **Algoritmo Genético (Genetic Algorithm, GA)**. O objetivo é explorar e comparar essas duas meta-heurísticas para resolver o TSP, um problema clássico de otimização combinatória.

---

## 📚 Descrição

O TSP consiste em encontrar a rota de menor distância que visita cada uma de \(n\) cidades exatamente uma vez e retorna à cidade inicial. Devido à sua complexidade (classificado como NP-difícil), métodos exatos tornam-se inviáveis para instâncias grandes, motivando o uso de meta-heurísticas como SA e GA.

---

### Metodologias
1. **Têmpera Simulada (SA)**:
   - Inspirada no processo físico de recozimento de metais.
   - Permite aceitar soluções piores em certas condições para escapar de mínimos locais.
   - Utiliza uma temperatura inicial adaptativa e um cronograma de resfriamento geométrico.

2. **Algoritmo Genético (GA)**:
   - Baseado em princípios evolutivos, como seleção, crossover e mutação.
   - Mantém uma população de soluções e utiliza busca local (2-opt) para refinamento.
   - Adapta dinamicamente a taxa de mutação ao longo das gerações.

---

### Implementação
- **Linguagem**: Python
- **Bibliotecas utilizadas**:
  - `NumPy`: Para operações matriciais e manipulação de arrays.
  - `Numba`: Para compilação JIT de funções críticas.
  - `CuPy`: Para aceleração por GPU (opcional).
  - `ThreadPoolExecutor`: Para paralelização de execuções independentes.

---

## 📊 Resultados

Os experimentos foram realizados utilizando instâncias da biblioteca TSPLIB. Os resultados mostram que:
- **SA**: Converge rapidamente para instâncias menores, mas apresenta limitações de escalabilidade.
- **GA**: Oferece melhor escalabilidade para problemas maiores, com soluções de alta qualidade.

| Técnica            | Complexidade                                                      | Escalabilidade         | Qualidade da Solução |
|--------------------|-------------------------------------------------------------------|------------------------|----------------------|
| Busca Cega         | \( \mathcal{O}(n!) \)                                             | Inviável (\(n > 20\))  | Exata                |
| Busca Informada    | \( O(n!) \)                                                       | Limitada               | Exata                |
| Têmpera Simulada   | \( \mathcal{O}(n^2 \cdot \text{iterações}) \)                     | Boa para \(n\) pequeno | Aproximada           |
| Algoritmo Genético | \( \mathcal{O}(\text{população} \cdot n \cdot \text{gerações}) \) | Boa para \(n\) grande  | Aproximada           |

---

## 📂 Estrutura do Projeto

```
travelling-salesman/
├── caixeiro-viajante.tex   # Documento LaTeX com a descrição do projeto
├── README.md               # Este arquivo
├── src/                    # Código-fonte em Python
│   ├── simulated_annealing.py
│   ├── genetic_algorithm.py
│   └── utils.py
├── data/                   # Instâncias do problema (TSPLIB)
│   ├── example1.tsp
│   └── example2.tsp
└── results/                # Resultados experimentais
    ├── sa_results.csv
    └── ga_results.csv
```

---

## 🔗 Artigo

O artigo está disponpivel em formato LaTeX na pasta `caixeiro-viajante/`.

---

## 🛠️ Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/iLukSbr/travelling-salesman.git
   cd travelling-salesman
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute os algoritmos:
     ```bash
     python main.py
     ```

4. Visualize os resultados na pasta `benchmark_results/`.
