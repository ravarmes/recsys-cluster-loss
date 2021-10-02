<h1 align="center">
    <img alt="RVM" src="https://github.com/ravarmes/recsys-antidote/blob/main/assets/logo.jpg" />
</h1>

<h3 align="center">
  Análise de agrupamento das perdas utilizando medidas de justiça individual
</h3>

<p align="center">Exemplo de agrupamentos utilizando medidas de justiça individual </p>

<p align="center">
  <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/ravarmes/recsys-cluster-loss?color=%2304D361">

  <a href="http://www.linkedin.com/in/rafael-vargas-mesquita">
    <img alt="Made by Rafael Vargas Mesquita" src="https://img.shields.io/badge/made%20by-Rafael%20Vargas%20Mesquita-%2304D361">
  </a>

  <img alt="License" src="https://img.shields.io/badge/license-MIT-%2304D361">

  <a href="https://github.com/ravarmes/recsys-cluster-loss/stargazers">
    <img alt="Stargazers" src="https://img.shields.io/github/stars/ravarmes/recsys-cluster-loss?style=social">
  </a>
</p>

<p align="center">
  <a href="#-sobre">Sobre o projeto</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#-links">Links</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#-licenca">Licença</a>
</p>

## :page_with_curl: Sobre o projeto <a name="-sobre"/></a>

> É proposta uma análise de agrupamento no contexto de sistemas de recomendação, considerando as perdas da medida de justiça individual.

O objetivo deste repositório é implementar os cálculos para uma análise de agrupamento no contexto de sistemas de recomendação, considerando as perdas da medida de justiça individual.

Os cálculos da justiça individual são baseados nas implementações do respositório [antidote-data-framework](https://github.com/rastegarpanah/antidote-data-framework) 

### Funções de Objetivo Social (Social Objective Functions)

```
* Individual fairness (Justiça Individual): a perda do usuário i é a estimativa do erro quadrático médio sobre as classificações conhecidas do usuário i
```

### Arquivos

```
* ArticleAntidoteData: implementação das medidas de justiça do usuário (ou funções de objetivo social)
* RecSysALS: implementação do sistema de recomendação baseado em filtragem colaborativa utilizando ALS (mínimos quadrados alternados)
* RecSysExampleData20Items: implementação de uma matriz de recomendações estimadas (apenas exemplo com valores aleatórios)
* TestArticleAntidoteData: arquivo para testar a implementação ArticleAntidoteData
```

## :link: Links <a name="-links"/></a>

- [Google Colaboratory](https://colab.research.google.com/drive/1aZIuljttlAaTq-LxtcXgjuBNnDCakuzE) - Notebook para demonstrar a utilização do algoritmo para uma base de dados pequena (40 usuários e 20 filmes);
- [Google Sheets](https://github.com/ravarmes/recsys-antidote/blob/main/docs/antidote-data-example.xlsx) - Planilha para demonstrar a utilização do algoritmo para uma base de dados pequena (40 usuários e 20 filmes);
- [Artigo](https://arxiv.org/pdf/1812.01504.pdf) - Fighting Fire with Fire: Using Antidote Data to Improve Polarization and Fairness of Recommender Systems;


## :memo: Licença <a name="-licenca"/></a>

Esse projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE.md) para mais detalhes.

## :email: Contato

Rafael Vargas Mesquita - [GitHub](https://github.com/ravarmes) - [LinkedIn](https://www.linkedin.com/in/rafael-vargas-mesquita) - [Lattes](http://lattes.cnpq.br/6616283627544820) - **ravarmes@hotmail.com**

---

Feito com ♥ by Rafael Vargas Mesquita :wink: