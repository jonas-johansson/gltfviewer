CC      ?= cc
CFLAGS  ?= -O2 -g
CFLAGS  += -std=c11 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -isystem third_party $(shell pkg-config --cflags sdl3)
LDLIBS  += $(shell pkg-config --libs sdl3) -lGL -lm

PREFIX  ?= $(HOME)/.local
BINDIR   = $(PREFIX)/bin
APPDIR   = $(PREFIX)/share/applications
ICONDIR  = $(PREFIX)/share/icons/hicolor/scalable/apps

gltfviewer: src/main.c third_party/cgltf.h third_party/stb_image.h third_party/stb_easy_font.h third_party/stb_image_write.h
	$(CC) $(CFLAGS) -o $@ src/main.c $(LDLIBS)

install: gltfviewer
	install -Dm755 gltfviewer $(BINDIR)/gltfviewer
	install -Dm644 data/gltfviewer.svg $(ICONDIR)/gltfviewer.svg
	sed 's|@BINDIR@|$(BINDIR)|' data/gltfviewer.desktop > $(APPDIR)/gltfviewer.desktop
	chmod 644 $(APPDIR)/gltfviewer.desktop
	-update-desktop-database $(APPDIR)
	-gtk-update-icon-cache -q -t $(PREFIX)/share/icons/hicolor
	xdg-mime default gltfviewer.desktop model/gltf+json model/gltf-binary

uninstall:
	rm -f $(BINDIR)/gltfviewer $(APPDIR)/gltfviewer.desktop $(ICONDIR)/gltfviewer.svg
	-update-desktop-database $(APPDIR)

clean:
	rm -f gltfviewer

.PHONY: install uninstall clean
