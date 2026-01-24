---
name: consultar-tcc
description: Consulta o documento do TCC antes de criar qualquer componente novo. Use quando precisar criar código, implementar funcionalidades, definir arquiteturas, ou tomar decisões técnicas. O TCC contém a proposta completa do trabalho e deve ser sempre referenciado para garantir alinhamento com os objetivos e metodologia definidos.
---

# Consultar TCC Antes de Criar

## Instruções Principais

**SEMPRE** consulte o documento do TCC antes de criar qualquer componente, funcionalidade, ou tomar decisões técnicas importantes.

### Quando Consultar

Consulte o TCC quando precisar:
- Criar novos módulos ou scripts
- Implementar funcionalidades
- Definir arquiteturas de modelos
- Escolher métricas ou metodologias
- Decidir sobre pré-processamento de dados
- Implementar validação ou testes
- Criar documentação técnica

### Localização do TCC

O documento completo do TCC está disponível em:
```
codigo/pipeline/others/generate_text_for_ia_by_pdf/Predição_Automática_de_Indicativos_Financeiros_para_Bolsa_de_Valores_Considerando_o_Aspecto_Temporal___Rafael_da_Silva_Melo___2025.txt
```

### Processo de Consulta

1. **Antes de criar algo novo**: Leia as seções relevantes do TCC
2. **Verifique alinhamento**: Certifique-se de que a criação está de acordo com:
   - Objetivos do trabalho (Seção 1.4)
   - Metodologia proposta (Capítulo 4)
   - Arquitetura definida (Seção 4.3)
   - Métricas de avaliação (Seção 4.5)
   - Fundamentação teórica (Capítulo 2)

3. **Mantenha consistência**: Garanta que novas implementações seguem os padrões e decisões já documentadas no TCC

### Seções Importantes do TCC

- **Capítulo 1**: Objetivos e justificativa
- **Capítulo 2**: Fundamentação teórica e conceitos
- **Capítulo 3**: Trabalhos relacionados e estado da arte
- **Capítulo 4**: Metodologia e próximos passos (CRÍTICO para implementação)
  - 4.1: Aquisição e descrição dos dados
  - 4.2: Pré-processamento e engenharia de atributos
  - 4.3: Arquitetura dos modelos
  - 4.4: Desenho experimental e treinamento
  - 4.5: Métricas de avaliação

### Exemplo de Uso

**Cenário**: Preciso criar um novo script de pré-processamento

**Ação**:
1. Ler a Seção 4.2 do TCC sobre pré-processamento
2. Verificar quais transformações são obrigatórias
3. Consultar a Seção 2.2 sobre estacionariedade e transformações
4. Criar o script seguindo exatamente o que foi definido no TCC

### Nota Importante

Este é o **TCC 2** (segunda parte do trabalho). O documento do TCC contém a proposta completa do TCC 1 que foi aprovada. Todas as implementações devem seguir rigorosamente o que foi proposto e aprovado no TCC 1.
