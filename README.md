# Motor de Visão IA - Detecção de Rompimento de Fel

Este projeto utiliza visão computacional para monitorar linhas de produção de frigoríficos e detectar manchas verdes (fel) em áreas proibidas (esteiras).

## Estrutura do Projeto
- `detector.py`: Lógica central de processamento de imagem e segmentação de cores.
- `main.py`: Servidor Flask que gerencia a conexão com a câmera e o streaming para o dashboard.
- `index.html`: Interface web moderna para monitoramento em tempo real.
- `requirements.txt`: Dependências do sistema.

## Como Iniciar
1. Execute o arquivo `setup.bat` para instalar as bibliotecas necessárias.
2. Edite o arquivo `main.py` e altere a variável `rtsp_url` para o endereço da sua câmera (ex: `rtsp://usuario:senha@ip:porta/stream`).
3. Execute o servidor: `python main.py`.
4. Abra o arquivo `index.html` no seu navegador.

## Próximos Passos
- [ ] Integrar com o bot de WhatsApp existente.
- [ ] Adicionar seletor visual de ROI (Região de Interesse) no dashboard.
- [ ] Configurar alertas sonoros na interface.
