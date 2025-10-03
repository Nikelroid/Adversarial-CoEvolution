import tkinter as tk
from tkinter import ttk
import pyperclip

class CardSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Card Selector")
        self.root.geometry("1000x700")
        
        # Card data
        self.suits = {'s': '♠', 'd': '♦', 'c': '♣', 'h': '♥'}
        self.ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        self.selected_cards = set()
        
        # Create UI
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title = tk.Label(self.root, text="Select Cards", font=("Arial", 18, "bold"))
        title.pack(pady=10)
        
        # Cards frame with scrollbar
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(canvas_frame, bg="white")
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="white")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create card buttons
        self.card_buttons = {}
        row = 0
        col = 0
        
        for suit_code, suit_symbol in self.suits.items():
            for rank in self.ranks:
                card_id = f"{rank}{suit_code}"
                
                # Determine color
                color = "red" if suit_code in ['h', 'd'] else "black"
                
                # Create frame for card
                card_frame = tk.Frame(scrollable_frame, relief=tk.RAISED, 
                                     borderwidth=3, bg="white", width=119, height=170)
                card_frame.grid(row=row, column=col, padx=8, pady=8)
                card_frame.grid_propagate(False)
                
                # Card display
                card_text = f"{rank}\n{suit_symbol}"
                
                card_label = tk.Label(card_frame, text=card_text, 
                                     font=("Arial", 34, "bold"), 
                                     fg=color, bg="white", cursor="hand2")
                card_label.pack(expand=True)
                
                # Bind click event
                card_label.bind("<Button-1>", lambda e, c=card_id, f=card_frame: self.toggle_card(c, f))
                card_frame.bind("<Button-1>", lambda e, c=card_id, f=card_frame: self.toggle_card(c, f))
                
                self.card_buttons[card_id] = card_frame
                
                col += 1
                if col >= 13:  # 13 cards per row
                    col = 0
                    row += 1
        
        # Generate button
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        generate_btn = tk.Button(btn_frame, text="Generate", 
                                font=("Arial", 14, "bold"),
                                bg="#4CAF50", fg="white",
                                padx=30, pady=10,
                                command=self.generate_output)
        generate_btn.pack()
        
        # Status label
        self.status_label = tk.Label(self.root, text="No cards selected", 
                                     font=("Arial", 11), fg="gray")
        self.status_label.pack(pady=5)
        
    def toggle_card(self, card_id, frame):
        if card_id in self.selected_cards:
            self.selected_cards.remove(card_id)
            frame.config(bg="white", relief=tk.RAISED)
            for widget in frame.winfo_children():
                widget.config(bg="white")
        else:
            self.selected_cards.add(card_id)
            frame.config(bg="#e3f2fd", relief=tk.SUNKEN)
            for widget in frame.winfo_children():
                widget.config(bg="#e3f2fd")
        
        # Update status
        count = len(self.selected_cards)
        self.status_label.config(text=f"{count} card{'s' if count != 1 else ''} selected")
    
    def generate_output(self):
        if not self.selected_cards:
            self.status_label.config(text="No cards selected!", fg="red")
            return
        
        # Convert to required format (A for Ace)
        output_cards = []
        for card in self.selected_cards:
            output_cards.append(card)
        
        # Sort cards
        output_cards.sort()
        output_text = ", ".join(output_cards)
        
        # Print to terminal
        print(output_text)
        
        # Copy to clipboard
        try:
            pyperclip.copy(output_text)
            self.status_label.config(text="Generated and copied to clipboard!", fg="green")
        except:
            self.status_label.config(text="Generated (clipboard not available)", fg="orange")
        
        # Reset selections
        self.reset_selections()
    
    def reset_selections(self):
        for card_id in list(self.selected_cards):
            frame = self.card_buttons[card_id]
            frame.config(bg="white", relief=tk.RAISED)
            for widget in frame.winfo_children():
                widget.config(bg="white")
        
        self.selected_cards.clear()

if __name__ == "__main__":
    root = tk.Tk()
    app = CardSelectorApp(root)
    root.mainloop()