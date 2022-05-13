# Flower-GAN
Création d'un GAN pour générer des images de fleurs

Plusieurs tentatives ont été mises en place. Le dataset est constitué d'environ 150 000 images de fleurs en 250x250.

## GAN classique

On a implémenté un GAN classique directement sur les images en 250x250, mais sans grand succès, le 'mode copllapse' est important lorsque le réseau converge

## GAN 64x64 suivi d'un upscaling

On a implémenté un GAN classique cett fois sur une version réduite des images du dataset à 64x64, puis de les upscaler avec un CNN. Si le GAN converge correctement, en revanche, l'upscaling ne fait que rendre les images plus lisses et elles restent flou. Probablement que l'upscaling par un facteur de plus de 4 était trop ambitieux. 
