<TeXmacs|2.1.5>

<style|generic>

<\body>
  <doc-data|<doc-title|Example of Kalman Filter (assignment
  1)>|<doc-author|<author-data|<author-name|Herbert Xin>>>>

  The MA process we have is

  <\equation*>
    y<rsub|t>=\<epsilon\><rsub|t>-\<theta\>\<epsilon\><rsub|t-1>
  </equation*>

  Write this into state-space model, we let
  <math|x<rsub|t>=<bmatrix|<tformat|<table|<row|<cell|\<epsilon\><rsub|t>>>|<row|<cell|\<epsilon\><rsub|t-1>>>>>>>,
  so\ 

  <\align*>
    <tformat|<table|<row|<cell|x<rsub|t+1>=>|<cell|A x<rsub|t>+C
    w<rsub|t+1>>>|<row|<cell|<bmatrix|<tformat|<table|<row|<cell|\<epsilon\><rsub|t+1>>>|<row|<cell|\<epsilon\><rsub|t>>>>>>=>|<cell|<bmatrix|<tformat|<table|<row|<cell|0>|<cell|0>>|<row|<cell|1>|<cell|0>>>>><bmatrix|<tformat|<table|<row|<cell|\<epsilon\><rsub|t>>>|<row|<cell|\<epsilon\><rsub|t-1>>>>>>+<bmatrix|<tformat|<table|<row|<cell|1>>|<row|<cell|0>>>>>\<epsilon\><rsub|t+1>>>>>
  </align*>

  Then\ 

  <\align*>
    <tformat|<table|<row|<cell|y<rsub|t>=>|<cell|G
    x<rsub|t>>>|<row|<cell|y<rsub|t>=>|<cell|<bmatrix|<tformat|<table|<row|<cell|1>>|<row|<cell|-\<theta\>>>>>>x<rsub|t>>>>>
  </align*>

  Initial belief on <math|\<epsilon\> >is N(0,1), so for our vector <math|x>,
  the prior is\ 

  <\equation*>
    x<rsub|t>\<sim\><with|font|cal|N><around*|(|\<mu\><rsub|x>,\<Sigma\><rsub|x>|)>
  </equation*>

  where <math|\<mu\><rsub|x>=<bmatrix|<tformat|<table|<row|<cell|0>>|<row|<cell|0>>>>>,\<Sigma\><rsub|x>=I<rsub|2>>,
  and the prior on y would be\ 

  <\equation*>
    y\<sim\><with|font|cal|N><around*|(|G\<mu\>,G\<Sigma\>G<rprime|'>|)>
  </equation*>

  Now <math|G\<mu\>=0>, and <math|G\<Sigma\>G<rprime|'>=<bmatrix|<tformat|<table|<row|<cell|1>|<cell|-\<theta\>>>>>><bmatrix|<tformat|<table|<row|<cell|1>|<cell|0>>|<row|<cell|0>|<cell|1>>>>><bmatrix|<tformat|<table|<row|<cell|1>>|<row|<cell|-\<theta\>>>>>>=1+\<theta\><rsup|2>>,
  so\ 

  <\equation*>
    y\<sim\><with|font|cal|N><around*|(|0,1<rsub|>+\<theta\><rsup|2>|)>
  </equation*>

  We term the prediction error <math|a>, so that\ 

  <\equation*>
    a\<equiv\>y-G<wide|x|^>\<sim\><with|font|cal|
    N><around*|(|0,G\<Sigma\>G<rprime|'>|)>
  </equation*>

  In our example

  <\equation*>
    a\<sim\><with|font|cal|N><around*|(|0,1<rsub|>+\<theta\><rsup|2>|)>
  </equation*>

  Since we don't observe the hidden state <math|x>, <math|<wide|x|^>> is our
  best prediction, or belief about <math|x>, and we use it to calculate our
  best belief of <math|y>, which is <math|<wide| y|^>>.

  Now we construct the relationship

  <\equation*>
    x<rsub|0>-<wide| x|^><rsub|0>=L<rsub|0>a<rsub|0>+\<eta\>
  </equation*>

  which means we believe there is an intrinic relationship between the
  prediction error and the deviation of belief on <math|x> from its true
  value.

  Muliply both side by <math|a<rsub|0<rprime|'>>> and take expectation:

  <\equation*>
    \<bbb-E\><around*|[|<around*|(|x<rsub|0>-<wide|x|^><rsub|0>|)>a<rsub|0><rprime|'>|]>=L<rsub|0>\<bbb-E\><around*|[|a<rsub|0>a<rsub|0><rprime|'>|]>
  </equation*>

  Now since the true LOM for y is\ 

  <\equation*>
    y<rsub|0>=G x<rsub|0>
  </equation*>

  then\ 

  <\equation*>
    a<rsub|0>=G<around*|(|x<rsub|0>-<wide|x|^><rsub|0>|)>
  </equation*>

  Plug this back into the equation earlier

  <\equation*>
    <around*|(|x<rsub|0>-<wide|x|^><rsub|0>|)><around*|(|x<rsub|0>-<wide|x|^><rsub|0>|)><rprime|'>G<rprime|'>=L<rsub|0><around*|(|1+\<theta\><rsup|2>|)>
  </equation*>

  Do some algebra gives:

  <\equation*>
    L<rsub|0>=G<rprime|'><around*|(|1+\<theta\><rsup|2>|)><rsup|-1>
  </equation*>

  So using our state-space model:

  <\align*>
    <tformat|<table|<row|<cell|x<rsub|1>=>|<cell|A<wide|x|^><rsub|0>+A<around*|(|x<rsub|0>-<wide|x|^><rsub|0>|)>+C
    w<rsub|t+1>>>|<row|<cell|<wide|x|^><rsub|1>=>|<cell|A<wide|x|^><rsub|0>+A
    L<rsub|0> a<rsub|0>=A<wide|x|^><rsub|0>+K<rsub|0> a<rsub|0>>>>>
  </align*>

  Subtracting these 2 equations give

  <\equation*>
    x<rsub|1>-<wide|x|^><rsub|1>=A<around*|(|x<rsub|0>-<wide|x|^><rsub|0>|)>+C
    w<rsub|t+1>-K<rsub|0> a<rsub|0>
  </equation*>

  Taking expectation of <math|<around*|(|x<rsub|1>-<wide|x|^><rsub|1>|)><around*|(|x<rsub|1>-<wide|x|^><rsub|1>|)><rprime|'>>
  gives

  <\equation*>
    \<Sigma\><rsub|1>=<around*|(|A-K<rsub|0>G|)>\<Sigma\><rsub|0><around*|(|A-K<rsub|0>G|)><rprime|'><rsub|>+C
    C<rprime|'>
  </equation*>
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>