---
layout: post
title: "Proof that the gradient points in the direction of steepest ascent"
keyword: "gradient ascent"
visible: 1
---

In this post, we are going to prove that the gradient of a function **points in the direction of steepest ascent**. Furthermore, we will show that the magnitude of the gradient vector is exactly the **rate of change in that direction**. 

To understand in throughly, we first need to cover a few preliminary topics: the law of cosines, a lemma that relates the dot product of 2 vectors to their lenghts and the angle between them, and directional derivatives.<br><br>

## Law of cosines

Suppose we have an arbitrary triangle as follows: 

{: .center}
<img src="/imgs/triangle.jpg" width="500" height="300">

We want to figure out the relationship between the sides \\(a\\), \\(b\\) and \\(c\\). If \\(\theta \\) were a right angle, then we could apply the Pythagorean theorem to obtain the well known relationship \\(c^2 = a^2 + b^2\\). However, our angle \\(\theta\\) is rather arbitrary. Nevertheless, it turns out that there exists a relationship between the sides of the triangle. It is called **the law of cosines**.

**Lemma 1.** If \\(a\\), \\(b\\) and \\(c\\) are the sides of a triangle and \\(\theta\\) is the angle between \\(a\\) and \\(b\\), then \\(c^2 = a^2 + b^2 - 2ab \cos \theta\\).

**Proof.** We would want to express \\(c\\) in terms the sides \\(a\\) and \\(b\\), and the angle between them, \\(\theta\\). To do this, we first express \\(m\\) and \\(e\\) in terms of these variables, and then use the Pythagorean theorem to derive an expression for \\(c\\).

First the expression for \\(e\\): \\[\cos(\theta) = \frac{d}{b} \implies d = b \cos \theta, \\\
e = a - d = a - b \cos \theta.\\]

Now, an expression for \\(m\\): \\(\sin \theta = \frac{m}{b} \implies m = b \sin \theta\\) 

Using the Pythagorean theorem to express \\(c\\) in terms of \\(a\\) and \\(b\\):
\\[c^2 = e^2 + m^2 = (a - b \cos \theta)^2 + (b \sin \theta)^2 = \\\ 
a^2 - 2 a b \cos \theta + b^2 \cos^2 \theta + b^2 \sin^2 \theta = \\\ 
a^2 - 2 a b \cos \theta + b^2 ((\cos^2 \theta + \sin^2 \theta) = \\\ 
a^2 + b^2 - 2 a b \cos \theta. \quad \blacksquare \\]

The formula we derived looks very much like the Pythagorean theorem, and indeed, the latter is the special case of the former. If \\(\theta = 90^{\circ} \\), \\(\cos \theta = 0\\) and our formula exactly simplifies to the Pythagorean theorem: \\(c^2 = a^2 + b^2\\).<br><br> 

## Dot product 

In this section we'll show that the dot product of two vectors is the product of their lengths and the cosine of the angle between them. 

To prove it, we make use of the fact that by definition, the angle between two vectors is the same as the angle between the sides of a triangle that corresponds to these vectors. This is illustrated in the following figure:

{: .center}
<img src="/imgs/triangles-dotprod.jpg" width="700" height="250">

**Lemma 2.** \\(a^T b = \lVert a \rVert \cdot \lVert b \rVert \cdot \cos(\theta)\\).

**Proof.** Using the law of cosines, we have that \\[\lVert a - b \rVert ^ 2  = \lVert a \rVert ^ 2 + \lVert b \rVert ^ 2 - 2 \lVert a \rVert \lVert b \rVert \cos \theta.\\]

Expanding the left hand side:  \\[\lVert a - b \rVert ^ 2 = (a - b)^T (a - b) = a^T a - a^T b - b^T a + b^T b = \\\ 
\lVert a \rVert ^ 2 - 2 a^T b + \lVert b \rVert ^ 2.\\]

Substituting this back to our initial relationship: \\[\lVert a \rVert ^ 2 - 2 a^T b + \lVert b \rVert ^ 2 = \lVert a \rVert ^ 2 + \lVert b \rVert ^ 2 - 2 \lVert a \rVert \lVert b \rVert \cos \theta \implies \\\
a^T b = \lVert a \rVert \lVert b \rVert \cos \theta. \quad \blacksquare \\]<br>

## Directional derivative

Suppose we have a multivariable function \\(f(x,y)\\) and we want to know how much does the value of \\(f\\) change when we move a little bit in the \\(x\\) direction (i.e. in the direction pointed to by the vector \\(\left[ {\begin{array}{c} 1&0 \end{array}} \right]^T \\)). This questioned is answered by the partial derivative  \\[\frac{\partial f(x, y)}{\partial x}.\\]

Similiarly, when moving a little bit in the \\(y\\) direction (i.e. in the direction pointed to by the vector \\(\left[ {\begin{array}{c} 0&1 \end{array}} \right]^T \\)), the resulting change in the function is given by \\[\frac{\partial f(x, y)}{\partial y}.\\]

However, what if we instead want to move in some other arbitrary direction, not necessarily directly along the \\(x\\) or \\(y\\) axis? For example, suppose we want to move southeast, which is given by the vector \\(\left[ {\begin{array}{c} 1&-1 \end{array}} \right]^T \\). It is important to notice that moving southeast can be expressed as moving 1 step along the \\(x\\)-axis and -1 steps along the \\(y\\)-axis. Indeed, any direction \\(v = \left[ {\begin{array}{c} a&b \end{array}} \right]^T \\) we would like to move can be expressed as moving \\(a\\) steps in the \\(x\\)-direction and \\(b\\) steps in the \\(y\\)-direction.

When viewed this way, it becomes obvious that when we want to know how much the value of \\(f\\) changes when we move in the \\(v = \left[ {\begin{array}{c} a&b \end{array}} \right]^T \\) direction is exactly how much \\(f\\) changes if we move \\(a\\) steps in the \\(x\\) direction and \\(b\\) steps in the y direction. And we already know how much the function changes when we move along the x or y directions a single step - this is given by the partial derivative! So we just need to multiply the partial derivatives by how many steps we need to take in the given direction. 

Thus, we can define the **directional derivative** of \\(f\\) along \\(v\\) at a point \\(\(x,y\)\\) as follows: \\[\frac{\partial f(x, y)}{\partial v} = a \frac{\partial f(x,y)}{\partial x} + b \frac{\partial f(x,y)}{\partial y} = \\\ \left[ {\begin{array}{c} a&b \end{array}} \right] \left[ {\begin{array}{c} \frac{\partial f(x,y)}{\partial x}&\frac{\partial f(x,y)}{\partial y} \end{array}} \right]^T = v^T \nabla f(x,y).\\]

One caveat with this definition is that if the magnitude of \\(v\\) increases by some constant, the value of the directional derivative would also increase by that same constant. To avoid this, we usually constrain the vector \\(v\\) to be of unit length, i.e. \\(\lVert v \rVert = 1\\).

Naturally, the previous discussion also holds for arbitrary vector \\(v\\) in higher dimensions.<br><br>

## Steepest ascent

Finally, we have all the tools to prove that the direction of steepest ascent of a function \\(f\\) at a point \\(\(x,y\)\\) (i.e. the direction in which \\(f\\) increases the fastest) is given by the gradient at that point \\(\(x,y\)\\).

We can express this mathematically as an optimization problem. Indeed, we want to find a vector \\(v^\*\\) such that when we move slightly in the direction pointed to by this vector, our function changes the most. If you think about it, this is equivalent to finding a vector \\(v^\*\\) such that the directional derivative of \\(f\\) along this vector is maximum:

\\[v^* = \underset{v, \lVert v \rVert = 1}{\arg \max} \ v^T \nabla f(x,y).\\]

Using Lemma 2, we can rewrite the dot product inside the argmax: 

\\[v^* = \underset{v, \lVert v \rVert = 1}{\arg \max} \ \lVert v \rVert \lVert \nabla f(x,y) \rVert \cos \theta. \\]

Because the length of the vector \\(v\\) is \\(1\\) and the gradient of \\(f\\) doesn't depend on \\(v\\), the relationship simplifies to 

\\[v^* = \underset{v, \lVert v \rVert = 1}{\arg \max} \ \cos \theta.\\]

Even though there is no explicit vector \\(v\\) in the optimization objective anymore, we still optimize over it, since the angle \\(\theta\\) obviously still depends on the vector \\(v\\). Because the \\(\cos\\) of an angle is maximizeed when the angle is exactly \\(0^{\circ}\\), we have that \\(v^\*\\) must be exactly parallel to \\(\nabla f(x,y)\\). In other words, **we have convinced ourselves that the direction of steepest ascent at a point \\(\(x,y\)\\), given by the vector \\(v^*\\), is indeed the same as the direction given by the gradient vector \\(\nabla f\\) at the point \\(\(x,y\)\\).**


It remains to be shown that the magnitude of the gradient \\(\nabla f(x,y)\\) is equal to the rate of change of the function when we move in the direction of steepest ascent. Since we know that the direction of steepest ascent is given by the vector \\(v^\*\\), we simply need to calculate the directional derivative \\(\frac{\partial f(x,y)}{\partial v^\*}\\) to find the greatest rate of change of \\(f\\) at the point \\(\(x, y\).\\)

The first thing we need is the vector \\(v^\*\\). Since \\(v^\*\\) is parallel to \\(\nabla f(x,y)\\), \\(v^\*\\) is some multiple \\(c\\) of the gradient vector, i.e. \\(v^\* = c \nabla f(x,y)\\). Luckily, having the constraint \\(\lVert v^\* \rVert = 1\\) enables us to figure out the exact value of the constant \\(c\\):
\\[\lVert v^\* \rVert = 1 \implies \lVert c \nabla f(x,y) \rVert = 1 \implies \lvert c \rvert \lVert \nabla f(x,y) \rVert = 1 \implies \\\ \lvert c \rvert = \frac{1}{\lVert \nabla f(x,y) \rVert} \implies c = \frac{1}{\lVert \nabla f(x,y) \rVert}.\\]

Substituting this \\(c\\) back into the relationship \\(v^\* = c \nabla f(x,y)\\), we obtain: \\[v^\* = \frac{\nabla f(x,y)}{\lVert \nabla f(x,y) \rVert}.\\]

Now that we know the exact value of \\(v^\*\\), we can calculate the directional derivative of \\(f\\) along \\(v^\*\\) at a point \\((x,y)\\):

\\[\frac{\partial f(x,y)}{\partial v^\*} = (v^\*)^T \nabla f(x,y) = \frac{(\nabla f(x,y))^T \nabla f}{\lVert \nabla f(x,y) \rVert} = \frac{\lVert \nabla f(x,y) \rVert ^ 2}{\lVert \nabla f(x,y) \rVert} = \lVert \nabla f(x,y) \rVert.\\]

So, **moving in the direction of steepest ascent (i.e. in the direction of the gradient), the rate at which the function changes is given by the magnitude of the gradient itself!**<br><br>


## Geometric intuition

In this post, we have mathematically proved that the direction of steepest ascent is given by the gradient. However, it may not be  intuitively clear why this should be the case. 

There is indeed another, more geometric way to see that the vector \\(v^\*\\) and \\(\nabla f\\) must be parallel. Minimizing the dot product between two vectors is the same as minimizing the projection length of one vector onto another. Viewed from this angle, we want to find a vector \\(v^\*\\) such that when it is projected onto \\(\nabla f\\), the length of this projection is maximized. Obviously, the projection length is maximized when the two vectors are exactly parallel. 
