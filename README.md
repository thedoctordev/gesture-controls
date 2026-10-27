# Gesture Controls

Aplicativo desktop experimental para cadastrar movimentos da mão pela webcam e
associá-los a ações no Ubuntu com GNOME/Wayland.

## Ações disponíveis

- Mover o cursor acompanhando a ponta do dedo indicador.
- Clique esquerdo, direito ou do meio.
- Pressionar uma tecla ou combinação (por exemplo `CTRL+ALT+T`).
- Minimizar, maximizar ou fechar a janela ativa.
- Mover a janela ativa: o gesto inicia o modo de movimentação do GNOME; mantenha
  a mão detectada e mova o indicador. Ao retirar a mão, o movimento é confirmado.

Os atalhos de janela usam os atalhos padrão do GNOME (`Alt+F9`, `Alt+F10`,
`Alt+F4` e `Alt+F7`). O resultado pode variar se esses atalhos foram alterados
nas configurações do desktop.

## Requisitos no Ubuntu 26.04

- Sessão GNOME com Wayland.
- Python 3.11 a 3.14 (compatível com o Python 3.14 esperado no Ubuntu 26.04).
- Webcam funcional.
- `ydotool` e seu daemon configurados para acessar `/dev/uinput`.

No Wayland, aplicativos comuns não podem controlar livremente o mouse e o
teclado de outros aplicativos. O `ydotool` injeta entrada pelo dispositivo
`uinput`; o daemon precisa estar ativo e com permissão de acesso ao dispositivo
e ao socket. Não execute o aplicativo inteiro como root. Consulte as instruções
do pacote da sua distribuição para configurar o daemon e as permissões.

## Instalação

Instale Python 3, o pacote `python3-venv` e `ydotool` pelos meios disponíveis para
sua instalação do Ubuntu, por exemplo:

```bash
sudo apt update
sudo apt install python3-venv ydotool
```

Na pasta do projeto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

O cliente `ydotool` depende do daemon `ydotoold`, que precisa de acesso a
`/dev/uinput`. Configure/ative o serviço fornecido pelo pacote do Ubuntu e
garanta que seu usuário possa acessar o socket do daemon. O nome e o modo de
ativação do serviço podem variar conforme o pacote; não execute a interface
gráfica como root. Em outro terminal com o ambiente virtual ativado, rode:

```bash
gesture-controls
```

Também é possível iniciar com `python -m gesture_controls`.

## Uso

1. Deixe a mão visível e bem iluminada diante da webcam.
2. Informe um nome, selecione a ação e, se necessário, o parâmetro.
3. Clique em **Gravar movimento do gesto (2 s)** e execute o movimento
   continuamente durante os dois segundos de gravação. O caminho precisa ocupar
   uma parte visível da imagem; uma mão parada não é cadastrada.
4. Repita para adicionar outros gestos. Um nome já cadastrado é atualizado.
5. Para remover um gesto, selecione-o na lista e clique em **Remover gesto
   selecionado**.

As trajetórias são salvas localmente em
`~/.config/gesture-controls/gestures.json`. A classificação compara o movimento
capturado com a trajetória cadastrada, normalizando posição inicial e escala e
tolerando variações de velocidade. Prefira movimentos com trajetórias bem
diferentes para ações diferentes.
As ações de clique, teclado e janela são disparadas uma vez por reconhecimento;
para repeti-las, retire a mão da câmera por um instante e reapresente o gesto.
O movimento do cursor usa a ponta do indicador enquanto a trajetória associada
é reconhecida. Para mover uma janela, mantenha o gesto de movimento reconhecido
e mova o indicador. Para atuar sobre uma janela, deixe-a ativa antes de executar
o gesto; a webcam deve permanecer ligada para a detecção. Cadastros antigos de
poses ficam marcados como **recadastre**: eles não contêm trajetória e precisam
ser gravados novamente para serem reconhecidos. Na primeira execução,
o modelo oficial de detecção de mãos do MediaPipe é baixado para
`~/.cache/gesture-controls/hand_landmarker.task`.

## Testes

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
