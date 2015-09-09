figure
plot(price_test,prices_SVM,'.b')
axis([4 22 4 22])
grid on

figure
plot(price_test,prices_pca_glmnet,'.m')
axis([4 22 4 22])
grid on

%Ideas: plot PCA score, try to explain why the low points look came out
%high? Look especially at low data points. Are those scores properly
%correlated? Where does it go wrong?
figure