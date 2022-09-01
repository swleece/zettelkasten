21

## Title
Measuring Purity in Decision tree models

## Description
Entropy as a measure of purity:
p1 = fraction of examples that are the label

H(p1) (entropy) is the range of purity, from 0 to 1

p0 = 1 - p1
H(p1) = -p1*log2(p1) - p0*log2(p0)
OR    = -p1*log2(p1) - (1 - p1)*log2(1 - p1)


## Additional Notes
note: 0 log(0) is undefined


H |
 1|        . .
  |     .       .
  |   .           .
  | .               .
  |.                 .
  |____________________
                      1
          p1


## Linked Cards
{{ direct link to another card }}

## Tags
[[ Machine Learning ]] 
