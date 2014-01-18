if &cp | set nocp | endif
let s:cpo_save=&cpo
set cpo&vim
inoremap <silent> <Plug>Tex_Completion :call Tex_Complete("default","text")
imap <silent> <Plug> <Nop>
inoremap <silent> <Plug>Tex_LeftRight =Tex_LeftRight()
inoremap <silent> <Plug>Tex_MathCal =Tex_MathCal()
inoremap <silent> <Plug>Tex_MathBF =Tex_MathBF()
inoremap <silent> <Plug>Tex_FastCommandChange :call Tex_ChangeCommand('no')
inoremap <silent> <Plug>Tex_FastCommandInsert =Tex_DoCommand('no')
inoremap <silent> <Plug>Tex_FastEnvironmentChange :call Tex_ChangeEnvironments()
inoremap <silent> <Plug>Tex_FastEnvironmentInsert =Tex_FastEnvironmentInsert("no")
imap <F1> <Plug>Tex_Help
imap <silent> <Plug>IMAP_JumpBack =IMAP_Jumpfunc('b', 0)
imap <silent> <Plug>IMAP_JumpForward =IMAP_Jumpfunc('', 0)
imap <silent> <F7> :exe ":ptag ".expand("<cword>")
imap <silent> <F6> :call GenTags()
imap <silent> <F4> :nohl
imap <silent> <F3> :silent make
imap <silent> <F2> :silent write
map! <S-Insert> <MiddleMouse>
nnoremap  :cprev 
noremap <NL> :bn 
noremap  :bp 
nnoremap  :cnext 
noremap   <PageDown>
vnoremap , :call ExecMap(',', 'v')
vnoremap ,pi :call VEnclose('', '', '\begin{picture}', '\end{picture}')
vnoremap ,mp :call VEnclose('', '', '\begin{minipage}', '\end{minipage}')
vnoremap <silent> ,lr :call VEnclose('\sbox{', '}', '\begin{lrbox}', '\end{lrbox}')
vnoremap ,fi :call VEnclose('', '', '\begin{figure}', '\end{figure}')
vnoremap ,pg :call VEnclose('\paragraph{', '}', '', '')
vnoremap ,s2 :call VEnclose('\subsubsection{', '}', '', '')
vnoremap ,ss :call VEnclose('\subsection{', '}', '', '')
vnoremap ,se :call VEnclose('\section{', '}', '', '')
vnoremap ,ch :call VEnclose('\chapter{', '}', '', '')
vnoremap ,pa :call VEnclose('\part{', '}', '', '')
vnoremap <silent> ,sl :call VEnclose('', '', '\begin{slide}', '\end{slide}')
vnoremap <silent> ,ov :call VEnclose('', '', '\begin{overlay}', '\end{overlay}')
vnoremap <silent> ,no :call VEnclose('', '', '\begin{note}', '\end{note}')
vnoremap <silent> ,tb :call VEnclose('', '', '\begin{thebibliography}', '\end{thebibliography}')
vnoremap <silent> ,ve :call VEnclose('', '', '\begin{verse}', '\end{verse}')
vnoremap <silent> ,vm :call VEnclose('\verb|', '|', '\begin{verbatim}', '\end{verbatim}')
vnoremap <silent> ,tp :call VEnclose('', '', '\begin{titlepage}', '\end{titlepage}')
vnoremap ,sp :call VEnclose('\subparagraph{', '}', '', '')
vnoremap <silent> ,qe :call VEnclose('', '', '\begin{quote}', '\end{quote}')
vnoremap <silent> ,qn :call VEnclose('', '', '\begin{quotation}', '\end{quotation}')
vnoremap <silent> ,fr :call VEnclose('{\raggedright ', '}', '\begin{flushright}', '\end{flushright}')
vnoremap <silent> ,fl :call VEnclose('', '', '\begin{flushleft}', '\end{flushleft}')
vnoremap <silent> ,fc :call VEnclose('', '', '\begin{filecontents}', '\end{filecontents}')
vnoremap <silent> ,do :call VEnclose('', '', '\begin{document}', '\end{document}')
vnoremap <silent> ,ce :call VEnclose('\centerline{', '}', '\begin{center}', '\end{center}')
vnoremap <silent> ,ap :call VEnclose('', '', '\begin{appendix}', '\end{appendix}')
vnoremap <silent> ,ab :call VEnclose('', '', '\begin{abstract}', '\end{abstract}')
vnoremap <silent> ,ma :call VEnclose('', '', '\begin{math}', '\end{math}')
vnoremap <silent> ,eq :call VEnclose('', '', '\begin{equation}', '\end{equation}')
vnoremap <silent> ,ea :call VEnclose('', '', '\begin{eqnarray}', '\end{eqnarray}')
vnoremap <silent> ,dm :call VEnclose('', '', '\begin{displaymath}', '\end{displaymath}')
vnoremap ,ar :call VEnclose('', '', '\begin{array}', '\end{array}')
vnoremap ,tr :call VEnclose('', '', '\begin{tabular}', '\end{tabular}')
vnoremap <silent> ,tg :call VEnclose('', '', '\begin{tabbing}', '\end{tabbing}')
vnoremap ,te :call VEnclose('', '', '\begin{table}', '\end{table}')
vnoremap <silent> ,tl :call VEnclose('', '', '\begin{trivlist}', '\end{trivlist}')
vnoremap <silent> ,ti :call VEnclose('', '', '\begin{theindex}', '\end{theindex}')
vnoremap <silent> ,it :call VEnclose('', '', '\begin{itemize}', '\end{itemize}')
vnoremap <silent> ,en :call VEnclose('', '', '\begin{enumerate}', '\end{enumerate}')
vnoremap ,de :call VEnclose('', '', '\begin{description}', '\end{description}')
vnoremap ,li :call VEnclose('', '', '\begin{list}', '\end{list}')
map ,F :set foldlevel=100
map ,f :set foldlevel=0
map ,e :e =expand("%:h")/
nmap K <Plug>ManPageView
vnoremap Q gq
nnoremap Q gqap
xmap S <Plug>VSurround
vmap [% [%m'gv``
map \mbt <Plug>TMiniBufExplorer
map \mbu <Plug>UMiniBufExplorer
map \mbc <Plug>CMiniBufExplorer
map \mbe <Plug>MiniBufExplorer
map \e :InlineEvaluate()
map \g :call GoogleThis()
map \o :call CopyCat() 
map \q :call QuickFixToggle()
vmap ]% ]%m'gv``
vnoremap ` :call ExecMap('`', 'v')
vnoremap <silent> `em :call VEnclose('\emph{', '}', '{\em', '\/}')
vnoremap <silent> `it :call VEnclose('\textit{', '}', '{\itshape ', '}')
vnoremap <silent> `sc :call VEnclose('\textsc{', '}', '{\scshape ', '}')
vnoremap <silent> `sl :call VEnclose('\textsl{', '}', '{\slshape ', '}')
vnoremap <silent> `up :call VEnclose('\textup{', '}', '{\upshape ', '}')
vnoremap <silent> `rm :call VEnclose('\textrm{', '}', '{\rmfamily ', '}')
vnoremap <silent> `sf :call VEnclose('\textsf{', '}', '{\sffamily ', '}')
vnoremap <silent> `tt :call VEnclose('\texttt{', '}', '{\ttfamily ', '}')
vnoremap <silent> `md :call VEnclose('\textmd{', '}', '{\mdseries ', '}')
vnoremap <silent> `bf :call VEnclose('\textbf{', '}', '{\bfseries ', '}')
vnoremap <silent> `$ :call VEnclose('$', '$', '\[', '\]')
vnoremap <silent> `{ :call VEnclose('\left\{ ', ' \right\}', '\left\{', '\right\}')
vnoremap <silent> `[ :call VEnclose('\left[ ', ' \right]', '\left[', '\right]')
vnoremap <silent> `( :call VEnclose('\left( ', ' \right)', '\left(', '\right)')
vmap a% [%v]%
nmap cs <Plug>Csurround
nmap ds <Plug>Dsurround
nmap gx <Plug>NetrwBrowseX
xmap gS <Plug>VgSurround
map j gj
map k gk
xnoremap <silent> s :echoerr 'surround.vim: Visual mode s has been removed in favor of S'
nmap ySS <Plug>YSsurround
nmap ySs <Plug>YSsurround
nmap yss <Plug>Yssurround
nmap yS <Plug>YSurround
nmap ys <Plug>Ysurround
nnoremap <silent> <Plug>Tex_LeftRight :call Tex_PutLeftRight()
vnoremap <silent> <Plug>Tex_MathCal `>a}`<i\mathcal{
vnoremap <silent> <Plug>Tex_MathBF `>a}`<i\mathbf{
nnoremap <Plug>Tex_RefreshFolds :call MakeTexFolds(1)
nnoremap <silent> <Plug>Tex_FastCommandChange :call Tex_ChangeCommand('no')
vnoremap <silent> <Plug>Tex_FastCommandInsert :call Tex_DoCommand('yes')
nnoremap <silent> <Plug>Tex_FastCommandInsert i=Tex_DoCommand('no')
nnoremap <silent> <Plug>Tex_FastEnvironmentChange :call Tex_ChangeEnvironments()
vnoremap <silent> <Plug>Tex_FastEnvironmentInsert :call Tex_FastEnvironmentInsert("yes")
nnoremap <silent> <Plug>Tex_FastEnvironmentInsert i=Tex_FastEnvironmentInsert("no")
nmap <F1> <Plug>Tex_Help
nmap <silent> <Plug> i
nnoremap <silent> <Plug>NetrwBrowseX :call netrw#NetrwBrowseX(expand("<cWORD>"),0)
vmap <silent> <Plug>IMAP_JumpBack `<i=IMAP_Jumpfunc('b', 0)
vmap <silent> <Plug>IMAP_JumpForward i=IMAP_Jumpfunc('', 0)
vmap <silent> <Plug>IMAP_DeleteAndJumpBack "_<Del>i=IMAP_Jumpfunc('b', 0)
vmap <silent> <Plug>IMAP_DeleteAndJumpForward "_<Del>i=IMAP_Jumpfunc('', 0)
nmap <silent> <Plug>IMAP_JumpBack i=IMAP_Jumpfunc('b', 0)
nmap <silent> <Plug>IMAP_JumpForward i=IMAP_Jumpfunc('', 0)
nmap <silent> <F7> :exe ":ptag ".expand("<cword>")
map <silent> <F6> :exe "!evince ". shellescape(expand("%:p:h:t")).".pdf &"
map <silent> <F4> :nohl
map <silent> <F3> :silent make :copen
map <silent> <F2> :silent write
noremap <BS> <PageUp>
noremap <Right> :bn
noremap <Down> :Tlist 
noremap <Left> :bp 
noremap <Up> :Explore 
map <S-Insert> <MiddleMouse>
imap S <Plug>ISurround
imap s <Plug>Isurround
imap <NL> <Plug>IMAP_JumpForward
imap  <Plug>Isurround
imap jj 
iabbr xuser =g:username 
iabbr ydate =strftime("%B %d, %Y") 
iabbr xdate =strftime("%d/%m/%y %H:%M:%S") 
let &cpo=s:cpo_save
unlet s:cpo_save
set autoindent
set background=dark
set backspace=indent,eol,start
set browsedir=current
set clipboard=autoselect,exclude:cons\\|linux,unnamed
set cmdheight=2
set complete=.,w,b,u,t,i,k
set confirm
set noequalalways
set expandtab
set fileencodings=ucs-bom,utf-8,default,latin1
set formatoptions=nrq1
set guicursor=a:blinkon0
set guifont=Inconsolata\ Medium\ 12
set guioptions=
set helplang=en
set hidden
set history=1000
set hlsearch
set ignorecase
set incsearch
set iskeyword=a-z,A-Z,_
set laststatus=2
set mouse=a
set pastetoggle=<F5>
set printoptions=paper:letter
set ruler
set runtimepath=~/.vim,~/.vim/bundle/syntastic,~/.vim/bundle/vim-fugitive,~/.vim/bundle/vim-pandoc,~/.vim/bundle/vim-surround,~/.vim/bundle/vim-template,/var/lib/vim/addons,/usr/share/vim/vimfiles,/usr/share/vim/vim74,/usr/share/vim/vimfiles/after,/var/lib/vim/addons/after,~/.vim/after
set scrolloff=10
set shiftwidth=2
set shortmess=atI
set showcmd
set showmatch
set smartcase
set statusline=%F%m%r%h%w\ [%Y]\ [%04l,%04v][%p%%]\ [LEN=%L]\ %{VimBuddy()}
set suffixes=.bak,~,.swp,.o,.info,.aux,.log,.dvi,.bbl,.blg,.brf,.cb,.ind,.idx,.ilg,.inx,.out,.toc
set tabstop=2
set termencoding=utf-8
set textwidth=72
set viminfo='100,<50,s10,h,!,n~/.viminfo
set visualbell
set whichwrap=b,s,<,>,h,l
set wildignore=*.bak,*.o,*.e,*~
set wildmenu
set wildmode=longest:full,full
set window=86
" vim: set ft=vim :
